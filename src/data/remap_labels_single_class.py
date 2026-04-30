"""Remap all YOLO class IDs to 0 for single-class training.

Copies label files from a source tree to a destination tree while replacing
the first token in each valid YOLO line by "0".
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap YOLO labels to a single class id 0.")
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Source root containing cam folders with txt labels.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Destination root for remapped labels.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="cam_1,cam_2,cam_3,cam_4",
        help="Comma-separated camera folders to process.",
    )
    return parser.parse_args()


def remap_file(src_path: Path, dst_path: Path) -> int:
    lines = src_path.read_text(encoding="utf-8").strip().splitlines()
    remapped: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[0] = "0"
            remapped.append(" ".join(parts))

    dst_path.write_text("\n".join(remapped) + "\n", encoding="utf-8")
    return len(remapped)


def main() -> None:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    cameras = [part.strip() for part in args.cameras.split(",") if part.strip()]

    total_files = 0
    total_lines = 0

    for cam in cameras:
        src_dir = src_root / cam
        dst_dir = dst_root / cam
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[WARN] Skip {cam}: source not found at {src_dir}")
            continue

        cam_files = 0
        for label_file in sorted(src_dir.glob("*.txt")):
            count = remap_file(label_file, dst_dir / label_file.name)
            cam_files += 1
            total_lines += count

        total_files += cam_files
        print(f"[OK] {cam}: {cam_files} label files remapped")

    print(f"[OK] Total files: {total_files}")
    print(f"[OK] Total label lines remapped: {total_lines}")
    print(f"[OK] Output root: {dst_root}")


if __name__ == "__main__":
    main()
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
