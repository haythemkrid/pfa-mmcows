"""
A2 — YOLO Format
Remap all cow IDs → class 0 (binary detection).
Symlink images, rewrite labels, generate dataset.yaml.
"""
from pathlib import Path
import pandas as pd

# === PATHS ===
INDEX_PATH = Path("/home/oussema/pfa-mmcows/store/data/clean/yolo_dataset_index.csv")
YOLO_ROOT = Path("/home/oussema/pfa-mmcows/store/data/clean/yolo_dataset")

# === LOAD INDEX ===
df = pd.read_csv(INDEX_PATH)

# === CREATE YOLO STRUCTURE ===
for split in ["train", "val", "test"]:
    (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# === SYMLINK IMAGES + REMAP LABELS ===
count = {"train": 0, "val": 0, "test": 0}

for _, row in df.iterrows():
    split = row["split"]
    cam = row["camera"]
    fname = Path(row["image_path"]).stem

    # Symlink image (saves disk)
    img_src = Path(row["image_path"])
    img_dst = YOLO_ROOT / "images" / split / f"{cam}_{fname}.jpg"
    if not img_dst.exists():
        img_dst.symlink_to(img_src)

    # Remap labels: cow_id → 0
    lbl_src = Path(row["label_path"])
    lbl_dst = YOLO_ROOT / "labels" / split / f"{cam}_{fname}.txt"
    if not lbl_dst.exists():
        with open(lbl_src) as f:
            lines = f.readlines()
        with open(lbl_dst, "w") as f:
            for line in lines:
                parts = line.strip().split()
                parts[0] = "0"
                f.write(" ".join(parts) + "\n")

    count[split] += 1

# === GENERATE YAML ===
yaml_content = f"""path: {YOLO_ROOT}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['cow']
"""
(YOLO_ROOT / "dataset.yaml").write_text(yaml_content)

# === REPORT ===
print("YOLO dataset built:")
for s, n in count.items():
    print(f"  {s}: {n} images")
print(f"\nYAML saved to {YOLO_ROOT / 'dataset.yaml'}")