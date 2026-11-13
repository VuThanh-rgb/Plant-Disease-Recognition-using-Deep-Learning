"""
config.py  -  MỤC 0: TOÀN BỘ THAM SỐ CỦA ĐỀ TÀI Ở MỘT CHỖ
=============================================================
Chỉ cần sửa file này (hoặc ghi đè trong notebook) để đổi thiết kế thí nghiệm.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------
    # 1) ĐƯỜNG DẪN
    # ------------------------------------------------------------------
    data_root: str = "data"        # chứa plantvillage/, plantdoc/, plantwild/
    output_dir: str = "outputs"    # checkpoint, bảng kết quả, hình vẽ

    # ------------------------------------------------------------------
    # 2) CHỌN DATASET  (xem README.md - mục "Quyết định dataset")
    # ------------------------------------------------------------------
    # PlantVillage luôn dùng làm nguồn huấn luyện (source domain).
    use_plantdoc: bool = True      # held-out test ngoài đồng ruộng (KHÔNG BAO GIỜ train)
    use_plantwild: bool = True     # dùng để fine-tune + test ảnh thực tế
    # Fine-tune trên dataset nào ở Giai đoạn 2?  "plantwild" | "plantdoc" | "none"
    finetune_on: str = "plantwild"
    # Đánh giá PlantDoc trên TOÀN BỘ ảnh (train+test) vì ta không train trên nó
    # -> ~2.5k ảnh thay vì ~230 ảnh của tập test chính thức => số liệu ổn định hơn.
    plantdoc_eval_on_all: bool = True

    # PlantWild phát hành 2 phiên bản (v1: 89 lớp có split + lá khỏe; v2: 115 lớp, không split,
    # không có lá khỏe). Ta dùng v1 => bỏ qua thư mục có tên chứa chuỗi này. Kiểm tra cấu trúc
    # thư mục sau khi tải (notebook mục 3) và chỉnh nếu tên khác.
    pw_exclude: tuple = ("v2",)

    # ------------------------------------------------------------------
    # 3) CHIA TẬP
    # ------------------------------------------------------------------
    pv_val_frac: float = 0.15
    pv_test_frac: float = 0.15
    # Nếu PlantWild tải về không kèm thư mục train/val/test thì tự chia
    # (xấp xỉ split chính thức của tác giả: ~70/10/20)
    pw_val_frac: float = 0.10
    pw_test_frac: float = 0.20
    seed: int = 42

    # ------------------------------------------------------------------
    # 4) MÔ HÌNH & HUẤN LUYỆN
    # ------------------------------------------------------------------
    models: tuple = ("resnet50", "efficientnet_b0", "mobilenetv3_large_100")
    pretrained: bool = True        # khởi tạo từ ImageNet (khuyến nghị)
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2           # Colab: 2; máy mạnh: 4-8
    amp: bool = True               # mixed precision (chỉ có tác dụng trên GPU)
    label_smoothing: float = 0.1
    weight_decay: float = 1e-2
    augmentation: str = "strong"   # "basic" | "strong"

    epochs_pv: int = 6             # Giai đoạn 1: train trên PlantVillage
    lr_pv: float = 3e-4
    epochs_ft: int = 8             # Giai đoạn 2: fine-tune trên ảnh thực tế
    lr_ft: float = 1e-4
    patience: int = 3              # early stopping theo val macro-F1

    # ------------------------------------------------------------------
    # 5) NGƯỠNG "NÊN DÙNG DATASET NÀO" (dùng trong analysis.coverage_report)
    # ------------------------------------------------------------------
    min_shared_classes: int = 10   # tối thiểu số lớp trùng với PlantVillage
    min_shared_images: int = 300   # tối thiểu số ảnh (thuộc các lớp trùng) để test đáng tin

    # ------------------------------------------------------------------
    # 6) GÁN NHÃN THỦ CÔNG (ghi đè khi bộ phân tích tự động hiểu sai tên lớp)
    #    {"tên thư mục gốc": "crop__condition"}; dùng None để loại hẳn lớp đó
    # ------------------------------------------------------------------
    manual_label_overrides: dict = None

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        self.manual_label_overrides = self.manual_label_overrides or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Đường dẫn con
    @property
    def pv_dir(self):
        return self.data_root / "plantvillage"

    @property
    def pd_dir(self):
        return self.data_root / "plantdoc"

    @property
    def pw_dir(self):
        return self.data_root / "plantwild"
