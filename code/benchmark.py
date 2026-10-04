"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1)."""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây (ms)."""
    # 1. Warmup: chạy bỏ đi >= 10 lần đầu để ổn định GPU clock và cuBLAS
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    times_ms = []

    # 2. Vòng lặp đo thực tế
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()

        fn()

        if sync is not None:
            sync()
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

    # 3. Tính toán các chỉ số thống kê bách phân (percentile)
    times_arr = np.array(times_ms)
    p50 = float(np.percentile(times_arr, 50))
    p95 = float(np.percentile(times_arr, 95))
    p99 = float(np.percentile(times_arr, 99))
    mean_val = float(np.mean(times_arr))

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean": mean_val,
        "n": iters
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    # Xác định hàm đồng bộ thiết bị
    sync_fn = torch.cuda.synchronize if target_device.type == "cuda" else None

    # Lấy thông tin thiết bị
    gpu_name = torch.cuda.get_device_name(0) if target_device.type == "cuda" else "CPU"

    # Tạo dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device)

    # Cấu hình kiểu dữ liệu (dtype)
    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()

    # Định nghĩa hàm forward tùy theo dtype
    if dtype == "amp" and target_device.type == "cuda":
        def run_forward():
            with torch.inference_mode(), torch.cuda.amp.autocast():
                _ = model(dummy_input)
    else:
        def run_forward():
            with torch.inference_mode():
                _ = model(dummy_input)

    # Gọi hàm bench đo đạc
    res = bench(run_forward, warmup=warmup, iters=iters, sync=sync_fn)

    # Tính thông lượng (throughput) - số ảnh xử lý được mỗi giây dựa trên p50
    images_per_s = float(batch_size / (res["p50"] / 1000.0)) if res["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
        "images_per_s": images_per_s,
        "torch": torch.__version__
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ thực tế khi chạy TTA K-views và so sánh với lý thuyết (K * p50)."""
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    sync_fn = torch.cuda.synchronize if target_device.type == "cuda" else None
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device)

    # Mô phỏng quá trình TTA: Chạy model K lần (hoặc flip + model K lần)
    def run_tta_forward():
        with torch.inference_mode():
            for i in range(k_views):
                # Ví dụ View 0 giữ nguyên, View 1 lật ngang
                if i % 2 == 1:
                    x = torch.flip(dummy_input, dims=[-1])
                else:
                    x = dummy_input
                _ = model(x)

    res_tta = bench(run_tta_forward, warmup=warmup, iters=iters, sync=sync_fn)
    
    # Đo đơn view để làm mốc so sánh
    res_single = latency_report(model, batch_size, img_size, dtype, device, warmup, iters)
    expected_p50 = res_single["p50"] * k_views

    res_tta["k_views"] = k_views
    res_tta["single_view_p50"] = res_single["p50"]
    res_tta["expected_p50"] = expected_p50
    res_tta["overhead_ratio"] = res_tta["p50"] / expected_p50 if expected_p50 > 0 else 1.0

    return res_tta


# --- UNIT TEST KIỂM TRA BENCHMARK ---
if __name__ == "__main__":
    print("=== Đang kiểm tra module Benchmark ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tạo mô hình ResNet18 đơn giản để test
    from torchvision.models import resnet18
    dummy_model = resnet18().to(device)

    # 1. Đo độ trễ đơn
    report = latency_report(dummy_model, batch_size=1, img_size=224, dtype="fp32", device=device, warmup=5, iters=20)
    print("Kết quả đo Batch 1 (FP32):")
    print(f" Thiết bị: {report['gpu']} | Torch: {report['torch']}")
    print(f" Latency p50: {report['p50']:.3f} ms | p95: {report['p95']:.3f} ms | p99: {report['p99']:.3f} ms")
    print(f" Thông lượng: {report['images_per_s']:.2f} img/s")

    # 2. Đo TTA (2 views)
    tta_report = tta_latency(dummy_model, k_views=2, batch_size=1, img_size=224, device=device, warmup=5, iters=20)
    print("\nKết quả đo TTA 2-views:")
    print(f" Thực tế p50: {tta_report['p50']:.3f} ms | Kỳ vọng (2 * 1-view): {tta_report['expected_p50']:.3f} ms")
    
    assert report["p50"] > 0, "Lỗi: p50 phải > 0!"
    print("\nUnit Test cho benchmark.py đã vượt qua thành công!")