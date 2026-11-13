"""
evaluate.py  -  MỤC 7: ĐÁNH GIÁ TRONG-MIỀN VÀ XUYÊN-DATASET
=============================================================
Với dataset đích (PlantDoc / PlantWild) chỉ một phần lớp trùng với PlantVillage, nên ta báo cáo
2 cách:
  * "restricted" : giới hạn dự đoán trong các lớp CÓ MẶT ở tập đích  (so sánh công bằng, dùng chính)
  * "full"       : để mô hình chọn trong cả 38 lớp                   (khắt khe hơn, tham khảo)
Chỉ số: accuracy, macro-F1  +  "generalization gap" = acc(PV test) - acc(đích).
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data_loading import make_loader


@torch.no_grad()
def predict_logits(model, loader, device, amp=True):
    model.eval().to(device)
    use_amp = amp and device.type == "cuda"
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def _scores(logits: torch.Tensor, labels: torch.Tensor, allowed: Optional[List[int]]):
    if allowed is not None:                      # che các lớp không có ở tập đích
        mask = torch.full_like(logits, float("-inf"))
        mask[:, allowed] = 0.0
        logits = logits + mask
    pred = logits.argmax(1).numpy()
    y = labels.numpy()
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0))
    return acc, f1, pred


def evaluate_set(model, df: pd.DataFrame, eval_tf, cfg, device, name: str) -> Dict:
    """df: ảnh đã có cột 'label' (>=0). Trả về dict số liệu + logits/labels để vẽ confusion matrix."""
    df = df[df["label"] >= 0]
    loader = make_loader(df, eval_tf, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    logits, labels = predict_logits(model, loader, device, cfg.amp)
    allowed = sorted(labels.unique().tolist())
    acc_r, f1_r, pred_r = _scores(logits, labels, allowed)
    acc_f, f1_f, _ = _scores(logits, labels, None)
    return {"set": name, "n_images": len(df), "n_classes": len(allowed),
            "acc": acc_r, "macro_f1": f1_r, "acc_full38": acc_f, "macro_f1_full38": f1_f,
            "_labels": labels.numpy(), "_pred": pred_r}


def results_table(rows: List[Dict], model_name: str, stage: str) -> pd.DataFrame:
    out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    out.insert(0, "stage", stage)
    out.insert(0, "model", model_name)
    return out


def add_generalization_gap(table: pd.DataFrame, in_domain_set: str = "PV-test") -> pd.DataFrame:
    """Thêm cột gap = acc(PV-test) - acc(tập đích) cho từng (model, stage)."""
    t = table.copy()
    base = t[t["set"] == in_domain_set].set_index(["model", "stage"])["acc"]
    t["gap_vs_PV"] = [round(base.get((m, s), np.nan) - a, 4) if st != in_domain_set else 0.0
                      for m, s, a, st in zip(t["model"], t["stage"], t["acc"], t["set"])]
    return t


def per_class_report(res: Dict, label_keys: List[str]) -> pd.DataFrame:
    """Precision/Recall/F1 từng lớp (chỉ các lớp có mặt)."""
    labs = np.unique(res["_labels"])
    rep = classification_report(res["_labels"], res["_pred"], labels=labs,
                                target_names=[label_keys[i] for i in labs],
                                output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T


def plot_confusion(res: Dict, label_keys: List[str], title: str, path=None, normalize=True):
    labs = np.unique(np.concatenate([res["_labels"], res["_pred"]]))
    cm = confusion_matrix(res["_labels"], res["_pred"], labels=labs).astype(float)
    if normalize:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(labs)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.42), max(5, n * 0.4)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    names = [label_keys[i] for i in labs]
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
