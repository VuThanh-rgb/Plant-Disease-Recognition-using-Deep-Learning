"""
download_data.py  -  MỤC 9: TẢI 3 DATASET
==========================================
Chạy trên Colab/Kaggle/máy có Internet. Mỗi hàm tải vào  data/<tên>/ .
Nếu một nguồn lỗi (link đổi, hết quota...) hãy tải tay rồi giải nén vào đúng thư mục:

    data/plantvillage/   ... (chứa thư mục color/<lớp>/*.jpg  hoặc <lớp>/*.jpg)
    data/plantdoc/       ... (chứa train/<lớp>/*.jpg và test/<lớp>/*.jpg)
    data/plantwild/      ... (chứa <lớp>/*.jpg, có thể có train/val/test)
"""
import subprocess
import zipfile
from pathlib import Path

from .data_index import has_images


def _run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _unzip_all(folder: Path):
    """Giải nén mọi file .zip trong folder (PlantWild phát hành dạng zip)."""
    for z in list(folder.rglob("*.zip")):
        print("Giải nén", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(z.parent)


def download_plantdoc(data_root="data"):
    """PlantDoc (~2.6k ảnh, 27 lớp, giấy phép CC BY 4.0) - repo GitHub bản phân loại."""
    dest = Path(data_root) / "plantdoc"
    if has_images(dest):
        print("PlantDoc đã có.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1",
          "https://github.com/pratikkayal/PlantDoc-Dataset", str(dest)])
    return dest


def download_plantwild(data_root="data"):
    """PlantWild (4.35GB, 18.5k ảnh v1 + 11.5k ảnh v2) từ Hugging Face.
    Giấy phép CC BY-NC-ND 4.0: chỉ dùng phi thương mại, KHÔNG phát hành lại bản đã chỉnh sửa."""
    from huggingface_hub import snapshot_download
    dest = Path(data_root) / "plantwild"
    snapshot_download(repo_id="uqtwei2/PlantWild", repo_type="dataset", local_dir=str(dest))
    _unzip_all(dest)
    return dest


def download_plantvillage_kaggle(data_root="data"):
    """PlantVillage từ Kaggle (cần ~/.kaggle/kaggle.json). Bản này có thư mục color/.
    Nếu slug lỗi: tải tay từ Kaggle/TFDS rồi đặt vào data/plantvillage/."""
    dest = Path(data_root) / "plantvillage"
    dest.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "datasets", "download", "-d", "abdallahalidev/plantvillage-dataset",
          "-p", str(dest), "--unzip"])
    return dest
