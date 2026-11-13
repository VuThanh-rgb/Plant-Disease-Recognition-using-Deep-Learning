"""
Kiểm thử không cần torch/GPU: ánh xạ nhãn, quét thư mục, chia tập, phủ lớp, phát hiện trùng ảnh.
Chạy:  python -m tests.test_labels_and_splits
"""
import shutil, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.label_space import make_key, add_keys, build_label_space, attach_labels, mapping_report
from src.data_index import scan_folder, find_plantvillage_root, ensure_splits, summarize_splits
from src.analysis import coverage_report, find_near_duplicates

PV = ["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
 "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
 "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
 "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
 "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
 "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
 "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___Late_blight",
 "Potato___healthy","Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
 "Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot","Tomato___Early_blight",
 "Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
 "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
 "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"]

PLANTDOC = {"Apple Scab Leaf":"apple__scab","Apple leaf":"apple__healthy","Apple rust leaf":"apple__rust",
 "Bell_pepper leaf":"pepper__healthy","Bell_pepper leaf spot":"pepper__bacterial_spot",
 "Blueberry leaf":"blueberry__healthy","Cherry leaf":"cherry__healthy","Corn Gray leaf spot":"corn__gray_leaf_spot",
 "Corn leaf blight":"corn__northern_leaf_blight","Corn rust leaf":"corn__rust","Peach leaf":"peach__healthy",
 "Potato leaf early blight":"potato__early_blight","Potato leaf late blight":"potato__late_blight",
 "Raspberry leaf":"raspberry__healthy","Soyabean leaf":"soybean__healthy",
 "Squash Powdery mildew leaf":"squash__powdery_mildew","Strawberry leaf":"strawberry__healthy",
 "Tomato Early blight leaf":"tomato__early_blight","Tomato Septoria leaf spot":"tomato__septoria",
 "Tomato leaf":"tomato__healthy","Tomato leaf bacterial spot":"tomato__bacterial_spot",
 "Tomato leaf late blight":"tomato__late_blight","Tomato leaf mosaic virus":"tomato__mosaic_virus",
 "Tomato leaf yellow virus":"tomato__yellow_leaf_curl","Tomato mold leaf":"tomato__leaf_mold",
 "Tomato two spotted spider mites leaf":"tomato__spider_mites","grape leaf":"grape__healthy",
 "grape leaf black rot":"grape__black_rot"}

def test_mapping():
    pv_keys = [make_key(c) for c in PV]
    assert None not in pv_keys, [c for c,k in zip(PV,pv_keys) if k is None]
    assert len(set(pv_keys)) == 38, "PlantVillage phải cho 38 khóa phân biệt"
    for raw, expected in PLANTDOC.items():
        got = make_key(raw)
        assert got == expected, (raw, got, expected)
    assert make_key("pineapple leaf") is None            # 'apple' không được khớp nhầm
    assert make_key("tomato weird canker disease") is None   # có dấu hiệu bệnh, không rõ -> loại
    assert make_key("x", {"x": "apple__scab"}) == "apple__scab"   # override thủ công
    print("OK mapping: 38 PV + 28 PlantDoc")

def _img(path, seed, size=40):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0,255,(size,size,3),dtype=np.uint8)).save(path)

def test_pipeline():
    tmp = Path(tempfile.mkdtemp())
    try:
        n = 0
        for c in PV:                                       # PlantVillage dạng .../color/<lớp>/
            for i in range(10): _img(tmp/"pv"/"Plant"/"color"/c/f"{i}.jpg", n); n += 1
        for c in PV[:2]:                                   # thư mục grayscale phải bị bỏ qua
            _img(tmp/"pv"/"Plant"/"grayscale"/c/"g.jpg", 999)
        for sp in ("train","test"):                        # PlantDoc dạng train/ test/
            for c in PLANTDOC:
                for i in range(6 if sp=="train" else 2): _img(tmp/"pd"/sp/c/f"{i}.jpg", n); n += 1
        for c in ["apple leaf","tomato early blight","banana sigatoka"]:   # PlantWild phẳng
            for i in range(12): _img(tmp/"pw"/c/f"{i}.jpg", n); n += 1

        pv = scan_folder(find_plantvillage_root(tmp/"pv"), "plantvillage")
        assert len(pv) == 38*10, len(pv)
        pd_ = scan_folder(tmp/"pd", "plantdoc"); pw = scan_folder(tmp/"pw", "plantwild")
        assert set(pd_.split) == {"train","test"} and pw.split.isna().all()

        pv, pd_, pw = [add_keys(d) for d in (pv, pd_, pw)]
        keys = build_label_space(pv); assert len(keys) == 38
        pv, pd_, pw = [attach_labels(d, keys) for d in (pv, pd_, pw)]
        assert (pw[pw.raw_label=="banana sigatoka"].label == -1).all()
        assert (pw[pw.raw_label=="apple leaf"].label >= 0).all()

        pv = ensure_splits(pv, .15, .15, 42); pd_ = ensure_splits(pd_, .15, 0., 42); pw = ensure_splits(pw, .1, .2, 42)
        assert set(pv.split) == {"train","val","test"} and "val" in set(pd_.split)
        # không rò rỉ: mỗi ảnh đúng 1 split
        assert pv.path.is_unique
        import pandas as pd
        print(summarize_splits(pd.concat([pv, pd_, pw])))
        print(coverage_report({"plantvillage": pv, "plantdoc": pd_, "plantwild": pw}, keys,
                              min_classes=10, min_images=300).to_string(index=False))
        print(mapping_report(pw, keys, "plantwild").to_string(index=False))

        # trùng ảnh: sao chép 1 ảnh PlantDoc vào "PlantWild" -> phải bị phát hiện
        src = pd_.iloc[0].path; dup = tmp/"pw"/"apple leaf"/"copy.jpg"; shutil.copy(src, dup)
        pw2 = scan_folder(tmp/"pw", "plantwild")
        d = find_near_duplicates(pd_, pw2, max_hamming=2)
        assert (d.path_a == src).any() and (d.path_b == str(dup)).any()
        print("OK pipeline + phát hiện ảnh trùng")
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    test_mapping(); test_pipeline(); print("TẤT CẢ TEST ĐẠT")
