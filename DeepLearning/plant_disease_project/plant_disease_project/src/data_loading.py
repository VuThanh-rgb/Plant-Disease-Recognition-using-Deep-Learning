"""
data_loading.py  -  MỤC 4: AUGMENTATION, DATASET, DATALOADER (PyTorch)
=======================================================================
"""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# 4.1  Augmentation
# ----------------------------------------------------------------------------
def build_transforms(img_size: int = 224, mode: str = "strong"):
    """Trả về (train_tf, eval_tf).
    - 'basic' : crop + lật ngang  (baseline)
    - 'strong': thêm xoay, đổi màu/sáng, blur, xóa ngẫu nhiên. PlantVillage có nền đơn giản
                nên augment mạnh giúp mô hình bớt "học thuộc nền" -> tổng quát hóa tốt hơn.
    """
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size), T.ToTensor(), norm])

    if mode == "basic":
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(), T.ToTensor(), norm])
    elif mode == "strong":
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(0.4, 0.4, 0.4, 0.08),
            T.RandomGrayscale(0.05),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(), norm,
            T.RandomErasing(p=0.25, scale=(0.02, 0.15))])
    else:
        raise ValueError("mode phải là 'basic' hoặc 'strong'")
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# 4.2  Dataset
# ----------------------------------------------------------------------------
class LeafDataset(Dataset):
    """Đọc ảnh từ DataFrame (cột 'path', 'label')."""

    def __init__(self, df: pd.DataFrame, transform, decode_size: int = 512):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform
        self.decode_size = decode_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        # PlantWild có ảnh tới 8256x5504: draft() giải mã JPEG ở độ phân giải thấp -> nhanh hơn nhiều
        img.draft("RGB", (self.decode_size, self.decode_size))
        img = img.convert("RGB")
        return self.transform(img), self.labels[i]


# ----------------------------------------------------------------------------
# 4.3  DataLoader
# ----------------------------------------------------------------------------
def make_loader(df: pd.DataFrame, transform, batch_size: int, shuffle: bool = False,
                num_workers: int = 2, balanced: bool = False) -> DataLoader:
    """balanced=True: lấy mẫu cân bằng theo lớp (dùng khi fine-tune trên tập mất cân bằng)."""
    ds = LeafDataset(df, transform)
    sampler = None
    if balanced:
        counts = df["label"].value_counts()
        weights = df["label"].map(lambda c: 1.0 / counts[c]).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(df), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0,
                      drop_last=False)
