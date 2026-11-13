"""
gradcam.py  -  MỤC 8: GIẢI THÍCH MÔ HÌNH BẰNG GRAD-CAM
=======================================================
Mục đích trong đề tài: kiểm tra mô hình huấn luyện trên PlantVillage nhìn vào VẾT BỆNH trên lá
hay "học tắt" từ nền/ánh sáng. Nếu heatmap nằm ở nền khi test ảnh ngoài đồng => giải thích được
vì sao điểm tổng quát hóa thấp.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data_loading import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts, self.grads = None, None
        self._handle = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        """Gỡ hook (tránh cộng dồn hook khi gọi nhiều lần)."""
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx=None):
        """x: (1,3,H,W). Trả về (heatmap HxW in [0,1], lớp dự đoán, độ tin cậy)."""
        self.model.zero_grad()
        logits = self.model(x)
        probs = logits.softmax(1)
        cls = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)                 # trọng số kênh
        cam = F.relu((w * self.acts).sum(1, keepdim=True))            # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls, float(probs[0, cls])


def _denorm(x: torch.Tensor) -> np.ndarray:
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def show_gradcam_grid(model, target_layer, df, eval_tf, label_keys, device,
                      n=8, seed=0, title="", path=None):
    """Vẽ lưới ảnh gốc + heatmap cho n ảnh ngẫu nhiên của df (cột path/label)."""
    cam_fn = GradCAM(model.to(device), target_layer)
    sample = df[df["label"] >= 0].sample(min(n, len(df)), random_state=seed)
    cols = min(4, len(sample))
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(3.2 * cols, 3.2 * rows * 2))
    axes = np.atleast_2d(axes)
    for k, (_, r) in enumerate(sample.iterrows()):
        img = Image.open(r["path"])
        img.draft("RGB", (512, 512))
        x = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
        cam, cls, conf = cam_fn(x)
        rgb = _denorm(x[0])
        i, j = divmod(k, cols)
        a1, a2 = axes[2 * i, j], axes[2 * i + 1, j]
        a1.imshow(rgb)
        a1.set_title(f"GT: {label_keys[r['label']]}", fontsize=7)
        a2.imshow(rgb)
        a2.imshow(cam, cmap="jet", alpha=0.45)
        ok = "✓" if cls == r["label"] else "✗"
        a2.set_title(f"{ok} Pred: {label_keys[cls]} ({conf:.0%})", fontsize=7,
                     color="green" if cls == r["label"] else "red")
    cam_fn.remove()
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
