"""
train.py  -  MỤC 6: VÒNG LẶP HUẤN LUYỆN (AMP, warmup + cosine, early stopping)
===============================================================================
fit() dùng cho CẢ HAI giai đoạn:
    Giai đoạn 1: train trên PlantVillage          (lr_pv, epochs_pv)
    Giai đoạn 2: fine-tune trên ảnh thực tế       (lr_ft, epochs_ft; head lr x10)
Checkpoint tốt nhất được chọn theo val macro-F1 (chịu mất cân bằng lớp tốt hơn accuracy).
"""
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):                  # torch cũ
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def _validate(model, loader, device, use_amp):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
        preds.append(out.argmax(1).cpu())
        labels.append(y)
    p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    acc = float((p == l).mean())
    f1 = float(f1_score(l, p, average="macro", labels=np.unique(l), zero_division=0))
    return acc, f1


def fit(model, train_loader, val_loader, *, epochs, lr, weight_decay=1e-2,
        label_smoothing=0.1, head_lr_mult=1.0, amp=True, patience=3,
        ckpt_path="best.pt", device=None, tag=""):
    """Huấn luyện và trả về (model đã nạp trọng số tốt nhất, DataFrame lịch sử)."""
    device = device or get_device()
    use_amp = amp and device.type == "cuda"
    model.to(device)

    # --- optimizer: head có thể dùng lr lớn hơn backbone (hữu ích khi fine-tune)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if id(p) in head_ids]
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": lr}, {"params": head, "lr": lr * head_lr_mult}],
        weight_decay=weight_decay)

    # --- scheduler theo từng bước: warmup 1 epoch rồi cosine về 0
    steps_per_epoch = len(train_loader)
    total, warm = epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = _make_scaler(use_amp)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    history, best_f1, bad = [], -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"{tag} ep{ep}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)

        val_acc, val_f1 = _validate(model, val_loader, device, use_amp)
        rec = dict(epoch=ep, train_loss=run_loss / seen, train_acc=correct / seen,
                   val_acc=val_acc, val_macro_f1=val_f1, seconds=time.time() - t0)
        history.append(rec)
        print(f"[{tag}] ep{ep:02d} loss={rec['train_loss']:.4f} tr_acc={rec['train_acc']:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({rec['seconds']:.0f}s)")

        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] early stopping (không cải thiện {patience} epoch)")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, pd.DataFrame(history)
