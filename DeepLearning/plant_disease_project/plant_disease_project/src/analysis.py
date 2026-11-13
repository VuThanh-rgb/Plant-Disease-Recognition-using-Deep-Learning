"""
analysis.py  -  MỤC 3: TÍNH TOÁN "NÊN DÙNG DATASET NÀO"  +  KIỂM TRA TRÙNG LẶP ẢNH
====================================================================================
(a) coverage_report : mỗi dataset phủ được bao nhiêu lớp / ảnh của không gian nhãn?
                      -> in ra kết luận USE / OPTIONAL / SKIP theo ngưỡng trong Config.
(b) find_near_duplicates : phát hiện ảnh trùng/na ná giữa 2 dataset (rò rỉ dữ liệu)
                           bằng dHash 64-bit (chỉ dùng PIL + numpy).
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image


# ----------------------------------------------------------------------------
# (a) Báo cáo độ phủ lớp
# ----------------------------------------------------------------------------
def coverage_report(dfs: Dict[str, pd.DataFrame], label_keys: List[str],
                    min_classes: int = 10, min_images: int = 300) -> pd.DataFrame:
    """dfs: {'plantvillage': df, 'plantdoc': df, ...}; df đã có cột key/label/split."""
    rows = []
    for name, df in dfs.items():
        shared = df[df["label"] >= 0]
        n_cls = shared["label"].nunique()
        rows.append({
            "dataset": name,
            "tổng ảnh": len(df),
            "số lớp gốc": df["raw_label"].nunique(),
            "lớp chung với PV": n_cls,
            "ảnh thuộc lớp chung": len(shared),
            "% ảnh dùng được": round(100 * len(shared) / max(len(df), 1), 1),
            "ảnh/lớp chung (trung vị)": int(shared.groupby("label").size().median()) if n_cls else 0,
        })
    rep = pd.DataFrame(rows)

    def verdict(r):
        if r["dataset"] == "plantvillage":
            return "USE (nguồn huấn luyện)"
        ok = r["lớp chung với PV"] >= min_classes and r["ảnh thuộc lớp chung"] >= min_images
        return "USE" if ok else "SKIP (độ phủ/số ảnh quá thấp để kết luận)"

    rep["kết luận"] = rep.apply(verdict, axis=1)
    return rep


# ----------------------------------------------------------------------------
# (b) Phát hiện ảnh trùng bằng dHash
# ----------------------------------------------------------------------------
def dhash(path: str, hash_size: int = 8) -> int:
    """Difference hash 64-bit: so sánh pixel kề nhau trên ảnh xám 9x8."""
    with Image.open(path) as im:
        im.draft("L", (128, 128))                 # giải mã JPEG nhanh với ảnh rất lớn
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def compute_hashes(paths: List[str]) -> np.ndarray:
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        try:
            out[i] = dhash(p)
        except Exception:
            out[i] = 0
    return out


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def find_near_duplicates(df_a: pd.DataFrame, df_b: pd.DataFrame, max_hamming: int = 4,
                         chunk: int = 200) -> pd.DataFrame:
    """Trả về các cặp (path_a, path_b, hamming) có khoảng cách Hamming <= max_hamming.
    Dùng để loại ảnh PlantDoc xuất hiện trong tập TRAIN của PlantWild (rò rỉ)."""
    ha, hb = compute_hashes(df_a["path"].tolist()), compute_hashes(df_b["path"].tolist())
    pa, pb = df_a["path"].to_numpy(), df_b["path"].to_numpy()
    pairs = []
    for s in range(0, len(ha), chunk):
        x = np.bitwise_xor(ha[s:s + chunk, None], hb[None, :])           # (chunk, Nb) uint64
        dist = _POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)    # popcount
        dist[(ha[s:s + chunk] == 0)[:, None] | (hb == 0)[None, :]] = 99   # bỏ ảnh lỗi (hash=0)
        ia, ib = np.where(dist <= max_hamming)
        pairs += [(pa[s + i], pb[j], int(dist[i, j])) for i, j in zip(ia, ib)]
    return pd.DataFrame(pairs, columns=["path_a", "path_b", "hamming"])
