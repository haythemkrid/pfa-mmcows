"""
Remap all YOLO label class IDs to 0 (single-class 'cow').

Reads from:  data/raw/visual_data/labels/combined/0725/cam_X/*.txt
Writes to:   data/raw/visual_data/labels/0725_single/cam_X/*.txt

Then updates the junction so YOLOv8's /images/→/labels/ lookup hits
the single-class labels instead.
"""
import os
from pathlib import Path

SRC_ROOT = Path(r"C:\Users\DELL\Desktop\test\mmcows\data\raw\visual_data\labels\combined\0725")
DST_ROOT = Path(r"C:\Users\DELL\Desktop\test\mmcows\data\raw\visual_data\labels\0725_single")

cams = ["cam_1", "cam_2", "cam_3", "cam_4"]
total = 0

for cam in cams:
    src_dir = SRC_ROOT / cam
    dst_dir = DST_ROOT / cam
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"  SKIP {cam} — source not found")
        continue

    count = 0
    for label_file in sorted(src_dir.glob("*.txt")):
        lines = label_file.read_text().strip().splitlines()
        remapped = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = "0"           # remap class ID to 0
                remapped.append(" ".join(parts))
        (dst_dir / label_file.name).write_text("\n".join(remapped) + "\n")
        count += 1

    print(f"  {cam}: {count} labels remapped")
    total += count

print(f"\n  Total: {total} label files written to {DST_ROOT}")
