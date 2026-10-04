"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md)."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model: nn.Module, loader, device: torch.device, view=None):
    """Chạy model trên loader ở chế độ eval và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []

    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device, non_blocking=True)

            # Áp dụng hàm biến đổi view nếu được cung cấp (ví dụ TTA)
            if view is not None:
                images = view(images)

            outputs = model(images)

            all_filenames.extend(filenames)
            all_targets.append(labels.numpy())
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)

    return all_filenames, y_true, logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên batch ảnh gốc."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch ảnh (N, C, H, W) trên chiều rộng (Width)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop_size: int = 224) -> list[torch.Tensor]:
    """Tạo 5 crops (4 góc + giữa) từ batch ảnh x."""
    _, _, H, W = x.shape
    assert H >= crop_size and W >= crop_size, "Kích thước ảnh phải lớn hơn hoặc bằng crop_size!"

    top_left = x[:, :, 0:crop_size, 0:crop_size]
    top_right = x[:, :, 0:crop_size, W - crop_size:W]
    bottom_left = x[:, :, H - crop_size:H, 0:crop_size]
    bottom_right = x[:, :, H - crop_size:H, W - crop_size:W]

    cy, cx = H // 2, W // 2
    half_crop = crop_size // 2
    center = x[:, :, cy - half_crop:cy + half_crop, cx - half_crop:cx + half_crop]

    return [top_left, top_right, bottom_left, bottom_right, center]


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch ảnh về các kích thước khác nhau (dùng cho các mạng CNN hỗ trợ Global Pooling)."""
    crops = []
    for s in sizes:
        resized = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        crops.append(resized)
    return crops


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy (view) của TTA thành một dự đoán xác suất (N, 9)."""
    if space == "prob":
        # Chuyển từng view qua softmax rồi tính trung bình cộng xác suất
        probs_per_view = [
            F.softmax(torch.from_numpy(logits), dim=-1).numpy()
            for logits in logits_per_view
        ]
        avg_probs = np.mean(probs_per_view, axis=0)
    elif space == "logit":
        # Tính trung bình cộng logit trước, sau đó mới qua softmax
        avg_logits = np.mean(logits_per_view, axis=0)
        avg_probs = F.softmax(torch.from_numpy(avg_logits), dim=-1).numpy()
    else:
        raise ValueError(f"Không hỗ trợ không gian gộp: {space}")

    return avg_probs


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Tính trung bình cộng xác suất của nhiều mô hình khác nhau (Ensemble)."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray | torch.Tensor, val_labels: np.ndarray | torch.Tensor) -> float:
    """Tìm nhiệt độ T > 0 bằng cách tối thiểu hóa NLL (Cross Entropy) trên tập VAL."""
    if isinstance(val_logits, np.ndarray):
        val_logits = torch.from_numpy(val_logits)
    if isinstance(val_labels, np.ndarray):
        val_labels = torch.from_numpy(val_labels)

    # Định nghĩa hàm loss NLL theo T
    def eval_loss(log_t):
        t = np.exp(log_t[0]) # Ép t luôn dương
        scaled_logits = val_logits / t
        loss = F.cross_entropy(scaled_logits, val_labels)
        return loss.item()

    # Tối ưu bằng thuật toán L-BFGS-B từ scipy
    res = minimize(eval_loss, x0=[0.0], method='L-BFGS-B')
    best_t = float(np.exp(res.x[0]))

    print(f"Khớp nhiệt độ Temperature Scaling thành công trên VAL: T = {best_t:.4f}")
    return best_t


def apply_temperature(logits: np.ndarray | torch.Tensor, T: float) -> np.ndarray:
    """Trả về xác suất softmax(logits / T) sau khi đã hiệu chuẩn nhiệt độ."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    scaled_logits = logits / max(1e-6, T)
    probs = F.softmax(scaled_logits, dim=-1).numpy()
    return probs


def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Hàm phụ trợ: Gộp một cặp (Conv2d, BatchNorm2d) thành một Conv2d duy nhất."""
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    # Chuẩn bị tham số W_conv và B_conv
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    b_conv = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)

    # Chuẩn bị tham số BatchNorm
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    invstd = gamma / torch.sqrt(var + eps)

    # Công thức gộp W' và b'
    w_fused = w_conv * invstd.reshape(-1, 1)
    b_fused = (b_conv - mean) * invstd + beta

    fused_conv.weight.copy_(w_fused.reshape(fused_conv.weight.shape))
    fused_conv.bias.copy_(b_fused)

    return fused_conv


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp tất cả các cặp BatchNorm2d vào Conv2d liền trước trong toàn bộ mô hình (chỉ áp dụng cho CNN)."""
    model.eval()
    model_fused = deepcopy(model)

    fused_count = 0
    # Duyệt qua các module con và thay thế nếu phát hiện cặp (Conv2d, BatchNorm2d)
    for name, child in model_fused.named_children():
        if len(list(child.children())) > 0:
            model_fused._modules[name] = fuse_conv_bn(child)
        else:
            # Kiểm tra cụm Sequential hoặc các block Conv2d + BatchNorm2d
            pass

    # Đơn giản hóa: Duyệt qua từng layer cụ thể nếu mô hình là chuỗi Sequential
    children = list(model_fused.named_children())
    for i in range(len(children) - 1):
        name1, module1 = children[i]
        name2, module2 = children[i + 1]
        if isinstance(module1, nn.Conv2d) and isinstance(module2, nn.BatchNorm2d):
            fused_conv = fuse_conv_bn_eval(module1, module2)
            setattr(model_fused, name1, fused_conv)
            setattr(model_fused, name2, nn.Identity())
            fused_count += 1

    print(f"Đã gộp thành công {fused_count} cặp Conv-BN trong mô hình.")
    return model_fused


# --- UNIT TEST KIỂM TRA HIỆU CHUẨN VÀ GỘP CONV-BN ---
if __name__ == "__main__":
    # 1. Test Temperature Scaling
    dummy_logits = np.random.randn(100, 9) * 2.0
    dummy_labels = np.random.randint(0, 9, size=(100,))
    T_fitted = fit_temperature(dummy_logits, dummy_labels)
    calibrated_probs = apply_temperature(dummy_logits, T_fitted)
    assert calibrated_probs.shape == (100, 9), "Lỗi output shape của Temperature Scaling!"

    # 2. Test Fuse Conv-BN
    conv_test = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False).eval()
    bn_test = nn.BatchNorm2d(16).eval()
    
    # Khởi tạo giá trị giả lập
    nn.init.standard_normal_(bn_test.running_mean)
    nn.init.uniform_(bn_test.running_var, 0.5, 1.5)

    x_test = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_original = bn_test(conv_test(x_test))
        fused_conv = fuse_conv_bn_eval(conv_test, bn_test)
        out_fused = fused_conv(x_test)

    max_diff = torch.max(torch.abs(out_original - out_fused)).item()
    print(f"Kiểm tra Fuse Conv-BN: Sai số lớn nhất giữa bản gốc và bản gộp = {max_diff:.8f}")
    assert max_diff < 1e-5, "Unit Test Fuse Conv-BN thất bại: Sai số gộp quá lớn!"
    print("Mọi Unit Test cho inference.py đã vượt qua thành công!")