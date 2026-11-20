"""
models.py  -  MỤC 5: TẠO MÔ HÌNH (timm) + CHỌN LỚP CHO GRAD-CAM
===================================================================
Các kiến trúc gợi ý (so sánh độ chính xác / tốc độ / độ tổng quát hóa):
    resnet50               : CNN kinh điển, baseline mạnh        (~25M tham số)
    efficientnet_b0        : cân bằng tốt độ chính xác / kích thước (~5M)
    mobilenetv3_large_100  : nhẹ, hợp triển khai trên điện thoại   (~5M)
    convnext_tiny          : CNN hiện đại (tùy chọn, nặng hơn)      (~28M)
"""
import timm
import torch
import torch.nn as nn


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_gradcam_layer(name: str, model: nn.Module) -> nn.Module:
    """Lớp tích chập cuối cùng (đầu ra còn thông tin không gian) cho Grad-CAM."""
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith(("efficientnet", "mobilenetv3")):
        return model.blocks[-1]
    if name.startswith("convnext"):
        return model.stages[-1]
    raise ValueError(f"Chưa khai báo lớp Grad-CAM cho '{name}'. Hãy thêm vào get_gradcam_layer().")


def load_weights(model: nn.Module, path, device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)
