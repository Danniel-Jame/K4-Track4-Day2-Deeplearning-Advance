"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        
    "mobilenetv3": "mobilenetv3_large_100",      
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp bằng thư viện timm."""
    
    # Xác định có dùng trọng số pretrained hay không
    use_pretrained = (init in ["frozen", "finetune"]) and pretrained
    
    # Tạo model từ timm, tự động thay thế classifier head cho phù hợp num_classes
    model = timm.create_model(
        name, 
        pretrained=use_pretrained, 
        num_classes=num_classes, 
        drop_rate=drop_rate
    )
    
    # Ghi log tag của trọng số để kiểm tra sau này
    if hasattr(model, 'pretrained_cfg') and model.pretrained_cfg:
        tag = model.pretrained_cfg.get('tag', 'unknown')
        print(f"Đã tải backbone: {name} | Pretrained Tag: {tag} | Init mode: {init}")
    
    if init == "frozen":
        freeze_backbone(model)
        
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head."""
    # Đóng băng toàn bộ model
    for param in model.parameters():
        param.requires_grad = False
        
    # Mở băng (unfreeze) riêng cho head
    # timm hỗ trợ hàm get_classifier() để lấy lớp cuối cùng (head) của hầu hết các kiến trúc
    classifier = model.get_classifier()
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True
    else:
        print("Cảnh báo: Không tự động tìm thấy classifier head để mở băng!")


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm để tối ưu hoá (slide Day 2, trang 52)."""
    
    # Lấy các tham số thuộc lớp Head
    head_params = set()
    classifier = model.get_classifier()
    if classifier is not None:
        head_params = set(classifier.parameters())
        
    group_backbone_decay = []
    group_backbone_no_decay = []
    group_head = []
    
    for param in model.parameters():
        if not param.requires_grad:
            continue
            
        if param in head_params:
            # Nhóm 3: Head mới
            group_head.append(param)
        else:
            # Tham số thuộc backbone
            if param.ndim > 1:
                # Nhóm 1: Trọng số của backbone (Conv, Linear weights)
                group_backbone_decay.append(param)
            else:
                # Nhóm 2: Norm và Bias của backbone (1D tensors) -> không dùng weight decay
                group_backbone_no_decay.append(param)
                
    return [
        {"params": group_backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": group_backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": group_head, "lr": lr_head, "weight_decay": weight_decay},
    ]


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Đếm GMAC (Giga Multiply-Accumulate Operations) bằng thư viện thop."""
    try:
        from thop import profile
        import warnings
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        
        # Tắt warning của thop để console sạch sẽ hơn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
            
        return macs / 1e9  # 1 GMAC = 1 tỷ MACs
        
    except ImportError:
        print("Vui lòng cài đặt thư viện 'thop' (pip install thop) để đếm GMACs.")
        return 0.0