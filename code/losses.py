"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls", "focal", "ce_weighted"."""
    kind = kind.lower()
    
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=kw.get("weight", None))
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        # Sử dụng tham số label_smoothing có sẵn tích hợp trong PyTorch 1.10+
        return nn.CrossEntropyLoss(weight=kw.get("weight", None), label_smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", kw.get("weight", None))
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        assert weight is not None, "Loại 'ce_weighted' yêu cầu truyền vào tensor 'weight'!"
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loại loss: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing (nhất quán với PyTorch implementation)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | list | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Tính log(p) và p
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        # Lấy p_t và log_p_t tương ứng với lớp target đúng
        log_p_t = log_p.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        p_t = p.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        
        # Tính hệ số điều chỉnh (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t
        
        # Áp dụng trọng số alpha_t theo từng lớp (nếu có)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(dim=0, index=target)
            loss = alpha_t * loss
            
        return loss.mean()


def class_weights(counts: list | np.ndarray | torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """Tính trọng số theo lớp dựa trên số lượng ảnh mẫu trong tập TRAIN."""
    counts = np.array(counts, dtype=np.float32)
    num_classes = len(counts)
    
    if beta <= 0.0:
        # Trọng số tỉ lệ nghịch đơn giản: w_c = 1 / n_c
        weights = 1.0 / counts
        # Chuẩn hoá về trung bình bằng 1
        weights = weights / np.mean(weights)
    else:
        # Class-balanced dựa theo "số mẫu hiệu dụng" (Cui et al.)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        # Chuẩn hoá tổng trọng số về số lượng lớp K
        weights = weights / np.sum(weights) * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def rand_bbox(size: tuple[int, ...], lam: float):
    """Hàm phụ trợ: Tạo bounding box ngẫu nhiên cho CutMix dựa trên tỷ lệ lambda."""
    W = size[2]
    H = size[3]
    
    # Tính kích thước khung cắt dựa vào lambda
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Lấy tâm ngẫu nhiên của khung cắt
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Giới hạn khung cắt trong diện tích ảnh
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    x_mixed = x.clone()

    if mode == "mixup":
        # Trộn tuyến tính các pixel ảnh
        x_mixed = lam * x + (1.0 - lam) * x[index]
    elif mode == "cutmix":
        # Cắt dán vùng không gian giữa hai ảnh
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Cập nhật chính xác lambda theo DIỆN TÍCH THỰC BỊ CẮT
        bbox_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = x.size(2) * x.size(3)
        lam = 1.0 - (bbox_area / float(total_area))
    else:
        raise ValueError(f"Không hỗ trợ chế độ trộn: {mode}")

    return x_mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính loss cho batch đã qua Mixup hoặc CutMix."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# --- UNIT TEST KIỂM TRA TÍNH ĐÚNG ĐẮN CỦA FOCAL LOSS ---
if __name__ == "__main__":
    # Test tự động kiểm tra gamma = 0 có bằng Cross-Entropy không (yêu cầu trong GUIDE.md)
    dummy_logits = torch.randn(10, 9)
    dummy_targets = torch.randint(0, 9, (10,))
    
    ce_loss = nn.CrossEntropyLoss()(dummy_logits, dummy_targets)
    focal_loss_g0 = FocalLoss(gamma=0.0)(dummy_logits, dummy_targets)
    
    diff = torch.abs(ce_loss - focal_loss_g0).item()
    print(f"Kiểm tra FocalLoss (gamma=0.0) vs CrossEntropyLoss: Chênh lệch = {diff:.8f}")
    assert diff < 1e-6, "Unit Test thất bại: Focal Loss với gamma=0 phải trùng khớp với CE Loss!"
    print("Unit Test thành công!")