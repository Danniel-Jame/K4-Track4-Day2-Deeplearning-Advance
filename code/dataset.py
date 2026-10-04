"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1)."""
    labels_dir = Path(labels_dir)
    
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train. In ra và trả về dict số liệu."""
    images_dir = Path(images_dir)
    
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    total_images = n_train + n_val + n_test
    
    # 3. Hợp ba tập phải bằng đúng 17.509 ảnh
    assert total_images == 17509, f"Lỗi: Tổng số ảnh phải là 17509, hiện tại là {total_images}"
    
    set_train = set(train_df['Filename'])
    set_val = set(val_df['Filename'])
    set_test = set(test_df['Filename'])
    
    # 2. Giao của từng cặp tập theo Filename phải RỖNG
    assert len(set_train.intersection(set_val)) == 0, "Lỗi: Tập train và val bị trùng lặp ảnh!"
    assert len(set_train.intersection(set_test)) == 0, "Lỗi: Tập train và test bị trùng lặp ảnh!"
    assert len(set_val.intersection(set_test)) == 0, "Lỗi: Tập val và test bị trùng lặp ảnh!"
    
    # 4. Mọi Filename đều tồn tại trong `images_dir`
    all_files = set_train.union(set_val).union(set_test)
    for filename in all_files:
        assert (images_dir / filename).exists(), f"Lỗi: Không tìm thấy ảnh {filename} trong thư mục."

    # 1. Số liệu chi tiết từng tập
    stats = {
        "total": total_images,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_per_class": train_df['Label'].value_counts().to_dict(),
        "val_per_class": val_df['Label'].value_counts().to_dict(),
        "test_per_class": test_df['Label'].value_counts().to_dict()
    }
    
    print(f"Kiểm tra dataset thành công! Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation."""
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    if not train:
        # Val/test: Resize 256 -> CenterCrop -> ToTensor -> Normalize
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize
        ])
        
    # Train cơ bản
    base_transforms = [
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        # Lưu ý: Không dùng RandomVerticalFlip cho thực vật có tính định hướng sinh trưởng trên xuống
    ]
    
    if aug == "color":
        base_transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    elif aug == "randaug":
        base_transforms.append(T.RandAugment())
        
    base_transforms.extend([T.ToTensor(), normalize])
    
    return T.Compose(base_transforms)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row['Filename']
        label = int(row['Label'])
        img_path = self.images_dir / filename
        
        # Mở bằng PIL và chuyển sang RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, filename


def seed_worker(worker_id):
    """Cố định seed cho DataLoader workers để đảm bảo khả năng tái lập."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader có hỗ trợ Oversampling nếu chỉ định `sampler="balanced"`."""
    dataset = DeepWeedsDataset(df, images_dir, transform)
    
    # Thiết lập Generator seed cho DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    
    loader_sampler = None
    shuffle = train
    drop_last = train
    
    # WeightedRandomSampler cho lớp mất cân bằng (GUIDE trục D)
    if train and sampler == "balanced":
        # Đếm số lượng của mỗi lớp
        class_counts = df['Label'].value_counts().sort_index().values
        # Trọng số của lớp tỉ lệ nghịch với số lượng mẫu
        class_weights = 1.0 / class_counts
        # Gán trọng số cho từng mẫu (sample) trong tập df
        sample_weights = [class_weights[label] for label in df['Label']]
        
        loader_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False # Bắt buộc False khi dùng custom sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=loader_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g
    )