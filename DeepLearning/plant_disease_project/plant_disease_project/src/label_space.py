"""
label_space.py  -  MỤC 1: THỐNG NHẤT KHÔNG GIAN NHÃN GIỮA CÁC DATASET
=======================================================================
Vấn đề: mỗi dataset đặt tên lớp một kiểu
    PlantVillage : "Tomato___Early_blight"
    PlantDoc     : "Tomato Early blight leaf"
    PlantWild    : (tên thư mục riêng của tác giả)

Giải pháp: đưa mọi tên về khóa chuẩn  "<cây>__<bệnh>"  (vd "tomato__early_blight").
Không gian nhãn của mô hình = các khóa chuẩn của PlantVillage (38 lớp).
Ảnh ở dataset khác chỉ được dùng khi khóa chuẩn của nó nằm trong không gian này
("lớp chung"). Phần còn lại bị loại và được in ra để bạn kiểm tra bằng mắt.

!!! Việc ánh xạ tự động dựa trên từ khóa. HÃY ĐỌC bảng in ra ở notebook (mục 4)
    và sửa  Config.manual_label_overrides  nếu thấy sai.
"""
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---- Từ khóa nhận diện LOẠI CÂY (thứ tự quan trọng) ------------------------
CROP_PATTERNS: List[Tuple[str, str]] = [
    ("pepper", r"\bpepper\b"),
    ("tomato", r"\btomato"),
    ("potato", r"\bpotato"),
    ("corn", r"\bcorn\b|\bmaize\b"),
    ("soybean", r"\bsoy"),
    ("strawberry", r"\bstrawberr"),
    ("raspberry", r"\braspberr"),
    ("blueberry", r"\bblueberr"),
    ("cherry", r"\bcherr"),
    ("peach", r"\bpeach"),
    ("apple", r"\bapple"),
    ("grape", r"\bgrape"),
    ("orange", r"\borange|\bcitrus"),
    ("squash", r"\bsquash"),
]

# ---- Từ khóa nhận diện BỆNH (thứ tự quan trọng: cụ thể -> chung) -----------
CONDITION_RULES: List[Tuple[str, str]] = [
    (r"healthy", "healthy"),
    (r"gray leaf spot|cercospora", "gray_leaf_spot"),
    (r"northern (corn )?leaf blight", "northern_leaf_blight"),
    (r"isariopsis|grape leaf blight", "grape_leaf_blight"),
    (r"early blight", "early_blight"),
    (r"late blight", "late_blight"),
    (r"esca|black measles", "esca"),
    (r"black rot", "black_rot"),
    (r"rust", "rust"),
    (r"scab", "scab"),
    (r"powdery mildew", "powdery_mildew"),
    (r"bacterial spot", "bacterial_spot"),
    (r"septoria", "septoria"),
    (r"leaf scorch", "leaf_scorch"),
    (r"mold|mould", "leaf_mold"),
    (r"spider mite|mites", "spider_mites"),
    (r"target spot", "target_spot"),
    (r"yellow leaf curl|yellow virus|yellow curl", "yellow_leaf_curl"),
    (r"mosaic", "mosaic_virus"),
    (r"greening|haunglongbing", "citrus_greening"),
]

# Nếu tên chứa các từ này mà không khớp luật nào => "unknown" (KHÔNG coi là khỏe)
DISEASE_HINT = re.compile(
    r"spot|blight|rot|rust|mildew|virus|mold|mould|scab|wilt|mite|canker|blast|"
    r"curl|scorch|mosaic|bacterial|fung|disease|esca|measles|greening|scald|gall|"
    r"anthracnose|spider|target|septoria|cercospora|haunglongbing|necrosis|streak"
)


def normalize(raw: str) -> str:
    """'Tomato___Early_blight' -> 'tomato early blight'."""
    text = raw.lower().replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_label(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Trả về (crop, condition). crop=None nếu không nhận ra cây;
    condition='unknown' nếu có dấu hiệu bệnh nhưng không có luật khớp."""
    text = normalize(raw)

    crop = None
    for name, pattern in CROP_PATTERNS:
        if re.search(pattern, text):
            crop = name
            break
    if crop is None:
        return None, None

    for pattern, cond in CONDITION_RULES:
        if re.search(pattern, text):
            return crop, cond

    # Luật theo từng loại cây (các tên mơ hồ)
    if crop == "corn" and "blight" in text:
        return crop, "northern_leaf_blight"
    if crop == "grape" and "blight" in text:
        return crop, "grape_leaf_blight"
    # GIẢ ĐỊNH: "Bell_pepper leaf spot" (PlantDoc) ~ bacterial spot (PlantVillage)
    if crop == "pepper" and "spot" in text:
        return crop, "bacterial_spot"

    if DISEASE_HINT.search(text):
        return crop, "unknown"
    return crop, "healthy"  # chỉ có tên cây ("Apple leaf") => lá khỏe


def make_key(raw: str, overrides: Optional[Dict[str, Optional[str]]] = None) -> Optional[str]:
    """Khóa chuẩn 'crop__condition' hoặc None nếu không ánh xạ được."""
    if overrides and raw in overrides:
        return overrides[raw]
    crop, cond = parse_label(raw)
    if crop is None or cond in (None, "unknown"):
        return None
    return f"{crop}__{cond}"


def add_keys(df: pd.DataFrame, overrides=None) -> pd.DataFrame:
    """Thêm cột 'key' vào DataFrame chỉ mục ảnh (dựa trên cột raw_label)."""
    df = df.copy()
    mapping = {r: make_key(r, overrides) for r in df["raw_label"].unique()}
    df["key"] = df["raw_label"].map(mapping)
    return df


def build_label_space(pv_df: pd.DataFrame) -> List[str]:
    """Không gian nhãn = tập khóa chuẩn của PlantVillage (đã sắp xếp)."""
    keys = sorted(pv_df["key"].dropna().unique())
    # Kiểm tra ánh xạ PlantVillage phải là đơn ánh (2 lớp gốc không trùng 1 khóa)
    dup = pv_df.dropna(subset=["key"]).groupby("key")["raw_label"].nunique()
    bad = dup[dup > 1]
    if len(bad):
        raise ValueError(f"Hai lớp PlantVillage bị gộp chung một khóa: {bad.to_dict()}")
    missing = pv_df[pv_df["key"].isna()]["raw_label"].unique()
    if len(missing):
        print(f"[CẢNH BÁO] Lớp PlantVillage không ánh xạ được: {list(missing)}")
    return keys


def attach_labels(df: pd.DataFrame, label_keys: List[str]) -> pd.DataFrame:
    """Thêm cột 'label' (int) cho ảnh có khóa nằm trong không gian nhãn; -1 nếu không."""
    index = {k: i for i, k in enumerate(label_keys)}
    df = df.copy()
    df["label"] = df["key"].map(index).fillna(-1).astype(int)
    return df


def mapping_report(df: pd.DataFrame, label_keys: List[str], name: str) -> pd.DataFrame:
    """Bảng: lớp gốc -> khóa chuẩn -> có thuộc không gian nhãn không? (để kiểm tra bằng mắt)"""
    g = df.groupby("raw_label").agg(n_images=("path", "count"), key=("key", "first")).reset_index()
    g["in_label_space"] = g["key"].isin(label_keys)
    g.insert(0, "dataset", name)
    return g.sort_values(["in_label_space", "raw_label"], ascending=[False, True]).reset_index(drop=True)
