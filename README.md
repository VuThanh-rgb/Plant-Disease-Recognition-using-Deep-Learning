# Nhận diện bệnh cây trồng từ ảnh lá + đánh giá tổng quát hóa đa dataset

## 1. Cấu trúc dự án

```
plant_disease_project/
├── README.md                    <- file này (kèm phân tích chọn dataset)
├── requirements.txt
├── notebooks/main.ipynb         <- chạy toàn bộ thí nghiệm, chia 14 mục / 38 cell
├── src/
│   ├── config.py                <- MỤC 0  mọi tham số
│   ├── label_space.py           <- MỤC 1  thống nhất nhãn giữa các dataset
│   ├── data_index.py            <- MỤC 2  quét thư mục, chia train/val/test (không cần torch)
│   ├── analysis.py              <- MỤC 3  báo cáo độ phủ lớp + phát hiện ảnh trùng
│   ├── data_loading.py          <- MỤC 4  augmentation, Dataset, DataLoader
│   ├── models.py                <- MỤC 5  ResNet50 / EfficientNet-B0 / MobileNetV3 (timm)
│   ├── train.py                 <- MỤC 6  vòng huấn luyện (AMP, cosine, early stopping)
│   ├── evaluate.py              <- MỤC 7  đánh giá trong-miền & xuyên-dataset, generalization gap
│   ├── gradcam.py               <- MỤC 8  giải thích bằng Grad-CAM
└── tests/test_labels_and_splits.py   <- kiểm thử không cần GPU
```

Cách chạy: tải 3 dataset **một lần** và đặt vào thư mục `data/`:

```
data/<bất kỳ>/plantvillage dataset/color/<lớp>/*.jpg   (PlantVillage)
data/<bất kỳ>/PlantDoc...  chứa train/ và test/        (PlantDoc - thư mục tên có chữ "plantdoc")
data/plantwild/...                                       (PlantWild, bỏ hoặc loại bản v2)
```
Code tự dò các thư mục này; nếu dò sai, điền `pv_override` / `pd_override` / `pw_override` trong `Config`.
Sau đó mở `notebooks/main.ipynb` (VS Code hoặc Colab có GPU) và chạy tuần tự từ trên xuống.

## 2. Quyết định dataset

### Số liệu nền (đã đối chiếu với trang dataset)

| Dataset | Ảnh | Lớp | Ghi chú |
|---|---|---|---|
| PlantVillage | ~54.3k | 38 | ảnh phòng thí nghiệm, nền đơn giản |
| PlantDoc | 2.598 | 27 (13 loài, 17 bệnh) | test chính thức chỉ ~231–236 ảnh |
| PlantWild v1 | 18.542 | 89 (56 bệnh + 33 lá khỏe) | có split chính thức 13.045 / 1.820 / 3.677 |
| PlantWild v2 | 11.488 | 115 (chỉ bệnh) | **không có split, không có lá khỏe** |

PlantWild: giấy phép **CC BY-NC-ND 4.0**, dung lượng ~4,35 GB.

### Kết luận: dùng CẢ BA, nhưng mỗi dataset một vai trò riêng

| Dataset | Vai trò | Lý do định lượng |
|---|---|---|
| **PlantVillage** | Huấn luyện nguồn (S1) + test trong miền | nhiều ảnh nhất, 38 lớp = không gian nhãn của mô hình |
| **PlantWild v1** | Fine-tune (train) + test ngoài miền (test chính thức) | đủ lớn để fine-tune; có split chính thức; có lá khỏe nên ánh xạ được vào 38 lớp |
| **PlantDoc** | **Chỉ test, không bao giờ train** | quá nhỏ để train (~8–10 ảnh/lớp ở tập test), nhưng độc lập với PlantWild |
| ~~PlantWild v2~~ | **Bỏ** | không có lá khỏe (không map được các lớp `healthy` của PlantVillage), không có split chính thức |

**Vì sao không trộn cả ba để train?** Khi đó không còn tập kiểm tra độc lập.

**Vì sao PlantDoc đánh giá trên toàn bộ ~2,5k ảnh thay vì ~230 ảnh test?** Vì ta không train trên nó. Sai số chuẩn (độ tin cậy 95%) của accuracy ~70%:

- 230 ảnh: ±6,0 điểm phần trăm
- 2.500 ảnh: ±1,8 điểm phần trăm

→ Dùng toàn bộ ảnh cho kết luận chặt hơn ~3 lần. (Notebook vẫn loại các ảnh PlantDoc gần trùng với ảnh PlantWild train/val bằng dHash, vì cả hai đều thu thập từ web.)

**Độ phủ lớp:** mô hình chỉ có thể được chấm trên các lớp trùng với 38 lớp của PlantVillage. Con số chính xác phụ thuộc cách ánh xạ tên lớp, nên **notebook Mục 5 tính ra bảng độ phủ thực tế** và tự kết luận `USE / SKIP` theo ngưỡng trong `Config` (≥10 lớp chung và ≥300 ảnh). Tôi chưa có danh sách lớp PlantWild để báo con số trước; bạn hãy đọc bảng ánh xạ ở Mục 4.

### Ba phương án

| Phương án | Dataset | Khi nào dùng |
|---|---|---|
| **A – Khuyến nghị** | PV + PlantWild v1 + PlantDoc | đủ thời gian/GPU; có đủ 3 mức: trong miền, ngoài miền đã thấy loại ảnh, ngoài miền hoàn toàn mới |
| **B – Tối thiểu** | PV + PlantDoc (`use_plantwild=False`, `finetune_on="none"`) | máy yếu / thiếu thời gian: chỉ có kết quả zero-shot, vẫn đủ chứng minh domain gap |
| C – Không khuyến nghị | trộn cả ba vào train | không có tập test độc lập |

Thời gian ước lượng thô cho phương án A trên Colab T4: khoảng 1,5–2,5 giờ (3 mô hình × 6 epoch ở S1 + fine-tune 1 mô hình). Đây là ước lượng, chưa đo thực tế; nút thắt thường là CPU đọc ảnh. Giảm `epochs_pv`, bỏ `resnet50`, hoặc đặt `FT_ALL=False` để rút ngắn.

## 3. Thiết kế thí nghiệm

1. **Không gian nhãn chung**: 38 khóa `cây__bệnh` của PlantVillage. Ảnh của dataset khác chỉ dùng khi tên lớp ánh xạ được (`label_space.py`).
2. **S1**: train trên PlantVillage → đo trên `PV-test`, `PlantWild-test`, `PlantDoc` (zero-shot).
3. **S2**: chọn kiến trúc tốt nhất bằng *val ảnh thực tế*, fine-tune trên PlantWild-train (lấy mẫu cân bằng lớp, lr thấp) → đo lại.
4. **Chỉ số**: accuracy, macro-F1 (cả khi giới hạn trong các lớp có mặt và khi dùng đủ 38 lớp), `gap_vs_PV`.
5. **Phân tích**: confusion matrix, báo cáo từng lớp, Grad-CAM trước/sau fine-tune, ablation augmentation (tùy chọn).
