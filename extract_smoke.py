"""
從 YPLOV11 資料集提取 smoke detected (class 4) 的圖片和標註
並轉換 label 為新資料集的 class 5（加在原本5類後面）
"""

import os
import shutil
from pathlib import Path

# YPLOV11 資料集中 smoke 的 class index
SMOKE_CLASS_IN_SOURCE = 4  # names: ['awake', 'distracted', 'eyes closed', 'phone usage', 'smoke detected', 'yawn']

# 在新合併資料集中，smoke 的 class index（原本5類之後）
SMOKE_CLASS_IN_TARGET = 5  # names: ['eyeclose', 'eyeopen', 'face', 'phone', 'yawn', 'smoke']

SOURCE_BASE = "C:/Users/Lin/Desktop/YPLOV11"
OUTPUT_BASE = "C:/Users/Lin/Desktop/smoke_only"

splits = ["train", "valid", "test"]

total_images = 0

for split in splits:
    img_src = Path(SOURCE_BASE) / split / "images"
    lbl_src = Path(SOURCE_BASE) / split / "labels"

    img_dst = Path(OUTPUT_BASE) / split / "images"
    lbl_dst = Path(OUTPUT_BASE) / split / "labels"

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    if not lbl_src.exists():
        print(f"[SKIP] {lbl_src} 不存在")
        continue

    count = 0
    for lbl_file in lbl_src.glob("*.txt"):
        with open(lbl_file, "r") as f:
            lines = f.readlines()

        # 找出含有 smoke 的行
        smoke_lines = [l for l in lines if l.strip().split()[0] == str(SMOKE_CLASS_IN_SOURCE)]

        if not smoke_lines:
            continue  # 這張圖沒有 smoke，跳過

        # 找對應的圖片
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = img_src / (lbl_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"[WARN] 找不到圖片: {lbl_file.stem}")
            continue

        # 複製圖片
        shutil.copy(img_file, img_dst / img_file.name)

        # 轉換 label：將 class index 改為目標 class index
        new_lines = []
        for line in smoke_lines:
            parts = line.strip().split()
            parts[0] = str(SMOKE_CLASS_IN_TARGET)  # 改 class index
            new_lines.append(" ".join(parts) + "\n")

        # 寫入新的 label 檔
        with open(lbl_dst / lbl_file.name, "w") as f:
            f.writelines(new_lines)

        count += 1

    print(f"[{split}] 提取了 {count} 張含抽菸的圖片")
    total_images += count

print(f"\n完成！共提取 {total_images} 張圖片到 {OUTPUT_BASE}")
