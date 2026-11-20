"""
data_index.py  -  MỤC 2: QUÉT THƯ MỤC ẢNH -> BẢNG CHỈ MỤC (DataFrame) + CHIA TẬP
==================================================================================
Mỗi dataset được biểu diễn bằng một DataFrame với các cột:
    path       : đường dẫn ảnh
    raw_label  : tên thư mục lớp gốc
    split      : 'train' | 'val' | 'test'
    dataset    : 'plantvillage' | 'plantdoc' | 'plantwild'
(+ 'key', 'label' được thêm ở label_space.py)

File này KHÔNG import torch -> chạy/kiểm thử nhanh, không cần GPU.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


# ----------------------------------------------------------------------------
# 2.1  Quét thư mục
# ----------------------------------------------------------------------------
def has_images(root) -> bool:
    """True nếu thư mục tồn tại và chứa ít nhất một ảnh."""
    root = Path(root)
    return root.exists() and any(p.suffix.lower() in IMG_EXT for p in root.rglob("*"))


def find_plantvillage_root(base: Path) -> Path:
    """Bản Kaggle của PlantVillage có 3 thư mục color/ grayscale/ segmented/.
    Ta chỉ dùng ảnh MÀU gốc (color)."""
    base = Path(base)
    for d in base.rglob("color"):
        if d.is_dir():
            return d
    return base


def scan_folder(root: Path, dataset: str, exclude: tuple = ()) -> pd.DataFrame:
    """Quét đệ quy: nhãn = tên thư mục cha của ảnh.
    Nếu một thư mục tổ tiên tên là train/val/test thì dùng làm split có sẵn.
    exclude: bỏ qua ảnh nằm trong thư mục có tên chứa chuỗi này (vd ("v2",) để bỏ PlantWild v2).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Không thấy thư mục dataset: {root}")
    rows = []
    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in IMG_EXT):
            continue
        parts = p.relative_to(root).parts
        if len(parts) < 2:        # ảnh nằm thẳng ở root, không có thư mục lớp
            continue
        if any(e.lower() in d.lower() for d in parts[:-1] for e in exclude):
            continue
        split: Optional[str] = None
        for part in parts[:-2]:   # tổ tiên (không tính thư mục lớp và tên file)
            split = SPLIT_ALIASES.get(part.lower(), split)
        rows.append({"path": str(p), "raw_label": p.parent.name,
                     "split": split, "dataset": dataset})
    if not rows:
        raise RuntimeError(f"Không tìm thấy ảnh nào trong {root}")
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2.2  Chia train/val/test (phân tầng theo lớp, có seed)
# ----------------------------------------------------------------------------
def _split_indices(idx: np.ndarray, rng: np.random.Generator, val_frac: float, test_frac: float):
    idx = rng.permutation(idx)
    n = len(idx)
    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    if n >= 3:  # lớp rất nhỏ vẫn có ít nhất 1 ảnh val/test (nếu frac > 0)
        n_test = max(n_test, 1) if test_frac > 0 else 0
        n_val = max(n_val, 1) if val_frac > 0 else 0
    return idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """Chia ngẫu nhiên phân tầng theo raw_label."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    df["split"] = "train"
    for _, grp in df.groupby("raw_label"):
        te, va, _ = _split_indices(grp.index.to_numpy(), rng, val_frac, test_frac)
        df.loc[te, "split"] = "test"
        df.loc[va, "split"] = "val"
    return df


def ensure_splits(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """- Nếu dataset đã có train & test chính thức: giữ nguyên, chỉ tách val từ train nếu thiếu.
       - Ngược lại: tự chia phân tầng."""
    have = set(df["split"].dropna().unique())
    if df["split"].notna().all() and {"train", "test"} <= have:
        if "val" in have:
            return df
        df = df.copy()
        rng = np.random.default_rng(seed)
        for _, grp in df[df["split"] == "train"].groupby("raw_label"):
            _, va, _ = _split_indices(grp.index.to_numpy(), rng, val_frac, 0.0)
            df.loc[va, "split"] = "val"
        return df
    return stratified_split(df, val_frac, test_frac, seed)


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="dataset", columns="split", values="path",
                          aggfunc="count", fill_value=0)
