"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F)."""
from __future__ import annotations

import argparse
from copy import deepcopy
import dataclasses
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo import được từ thư mục hiện tại/starter
import dataset
import losses
import model as model_lib

# Import các hàm từ eval.py gốc theo quy định bài lab
try:
    from eval import compute_metrics, save_predictions
except ImportError:
    # Trường hợp chạy độc lập không nằm cạnh eval.py
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval import compute_metrics, save_predictions


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint...
    pred_dir: str = "predictions"     # file dự đoán nộp bài
    curves_dir: str = "curves"         # lưu ảnh biểu đồ đường cong
    # --- chỉ bật ở Bước 4 (chung kết) ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn file dự đoán chuẩn: <pred_dir>/<exp_id>_seed<k>_<split>.csv ."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định seed ngẫu nhiên cho tính tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Tạo AdamW với 3 nhóm tham số (theo model.param_groups)."""
    groups = model_lib.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Lịch trình Warmup tuyến tính + Cosine decay theo từng bước (step)."""
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    total_steps = cfg.epochs * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential Moving Average duy trì trọng số trung bình động."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Huấn luyện 1 epoch."""
    model.train()
    if cfg.init == "frozen":
        model_lib.freeze_backbone(model)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.eval()

    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Áp dụng Mixup hoặc CutMix nếu cấu hình yêu cầu
        if cfg.mix and cfg.mix in ["mixup", "cutmix"]:
            images, targets = losses.mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                outputs = model(images)
                loss = losses.mixed_loss(criterion, outputs, targets)
        else:
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

        if cfg.amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": total_loss / max(1, total_samples), "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Đánh giá mô hình trên DataLoader không tính gradient."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_filenames.extend(filenames)
            all_targets.append(labels.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, total_samples)

    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong Loss & Macro-F1 theo epoch."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_loss, 'r--', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val Macro-F1', color=color)
    ax2.plot(epochs, val_f1, 'b-o', label='Val Macro-F1')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: Config) -> dict:
    """Cấu trúc huấn luyện pipeline hoàn chỉnh."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_transform = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_transform = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_transform, cfg.batch_size,
        train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_transform, cfg.batch_size,
        train=False, num_workers=cfg.num_workers
    )

    model = model_lib.build_model(
        cfg.backbone, pretrained=True, num_classes=dataset.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init
    ).to(device)

    weights = None
    if cfg.class_weight_beta is not None:
        train_counts = train_df['Label'].value_counts().sort_index().values
        weights = losses.class_weights(train_counts, beta=cfg.class_weight_beta).to(device)

    criterion = losses.build_criterion(
        kind=cfg.loss, smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma, weight=weights
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp) if cfg.amp else None
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = r_dir / "best_model.pth"
    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        if ema is not None:
            ema.apply_shadow(model)

        val_filenames, val_true, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        val_probs = F.softmax(torch.tensor(val_logits), dim=-1).numpy()
        val_preds = np.argmax(val_probs, axis=1)

        # Tránh KeyError: Lấy an toàn hoặc tự tính thủ công Top-1 Accuracy
        val_top1_acc = float(np.mean(val_preds == val_true))
        
        try:
            val_eval_metrics = compute_metrics(val_true, val_preds)
            val_macro_f1 = val_eval_metrics.get("macro_f1", 0.0)
        except Exception:
            # Fallback nếu hàm compute_metrics yêu cầu signature khác (vd: thêm val_probs)
            val_eval_metrics = compute_metrics(val_true, val_preds, val_probs)
            val_macro_f1 = val_eval_metrics.get("macro_f1", 0.0)

        if ema is not None:
            ema.restore(model)

        rec = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_loss,
            "val_top1": val_top1_acc,
            "val_macro_f1": val_macro_f1,
            "lr": train_metrics["lr"],
            "time_sec": t_epoch
        }
        history.append(rec)

        print(f"[{cfg.exp_id}|Epoch {epoch:02d}/{cfg.epochs:02d}] "
              f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_macro_f1:.4f} | "
              f"Time: {t_epoch:.1f}s")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            save_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "macro_f1": val_macro_f1,
                "cfg": dataclasses.asdict(cfg)
            }
            if ema is not None:
                save_dict["ema_shadow"] = ema.shadow
            torch.save(save_dict, best_ckpt_path)

    ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    if ema is not None and "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
        ema.apply_shadow(model)

    val_filenames, val_true, val_logits, _ = evaluate(model, val_loader, criterion, device)
    val_probs = F.softmax(torch.tensor(val_logits), dim=-1).numpy()
    save_predictions(pred_path(cfg, "val"), val_filenames, val_true, val_probs)

    test_macro_f1 = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, val_transform, cfg.batch_size,
            train=False, num_workers=cfg.num_workers
        )
        test_filenames, test_true, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = F.softmax(torch.tensor(test_logits), dim=-1).numpy()
        test_preds = np.argmax(test_probs, axis=1)
        save_predictions(pred_path(cfg, "test"), test_filenames, test_true, test_probs)
        
        try:
            test_eval_metrics = compute_metrics(test_true, test_preds)
            test_macro_f1 = test_eval_metrics.get("macro_f1", 0.0)
        except Exception:
            test_eval_metrics = compute_metrics(test_true, test_preds, test_probs)
            test_macro_f1 = test_eval_metrics.get("macro_f1", 0.0)

    pd.DataFrame(history).to_csv(r_dir / "history.csv", index=False)
    plot_curves(
        history,
        Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png",
        title=f"{cfg.exp_id} ({cfg.backbone}) - Best F1: {best_macro_f1:.4f} (Ep {best_epoch})"
    )

    n_params = model_lib.count_params(model)
    gmacs = model_lib.count_gmacs(model, cfg.img_size)

    return {
        "exp_id": cfg.exp_id,
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "test_macro_f1": test_macro_f1,
        "avg_epoch_time": float(np.mean(epoch_times)),
        "params_m": n_params,
        "gmacs": gmacs
    }


def parse_overrides(pairs: list[str]) -> dict:
    res = {}
    field_types = {f.name: f.type for f in dataclasses.fields(Config)}

    for pair in pairs:
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k not in field_types:
            raise KeyError(f"Trường '{k}' không thuộc Config!")

        target_type = field_types[k]

        if v.lower() == "none":
            res[k] = None
        elif v.lower() == "true":
            res[k] = True
        elif v.lower() == "false":
            res[k] = False
        elif target_type in [int, "int"]:
            res[k] = int(v)
        elif target_type in [float, "float", "float | None"]:
            res[k] = float(v)
        else:
            res[k] = v
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy huấn luyện mô hình DeepWeeds.")
    parser.add_argument("--set", nargs="*", default=[], help="Cú pháp: --set exp_id=B01 backbone=resnet50 seed=0")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)

    print(f"=== Bắt đầu chạy thí nghiệm {cfg.exp_id} | Seed: {cfg.seed} | Backbone: {cfg.backbone} ===")
    summary = run(cfg)
    print("=== Hoàn thành thành công ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()