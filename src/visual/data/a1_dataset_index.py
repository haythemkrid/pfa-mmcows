"""
A1 — Dataset Index
Build master DataFrame from all 4 cameras + 70/15/15 temporal split.
No files copied — all paths point to originals.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# === PATHS ===
DATA_ROOT = Path("/home/oussema/pfa-mmcows/store/data/raw/visual_data")
IMAGES = DATA_ROOT / "images" / "0725"
LABELS = DATA_ROOT / "labels" / "combined" / "0725"
CAMERAS = ["cam_1", "cam_2", "cam_3", "cam_4"]
OUTPUT = Path("/home/oussema/pfa-mmcows/store/data/clean/yolo_dataset_index.csv")

# === BUILD INDEX ===
rows = []
for cam in CAMERAS:
    for img_path in sorted((IMAGES / cam).glob("*.jpg")):
        lbl_path = LABELS / cam / (img_path.stem + ".txt")
        rows.append({
            "camera": cam,
            "timestamp": img_path.stem.split("_")[0],
            "image_path": str(img_path),
            "label_path": str(lbl_path),
            "has_label": lbl_path.exists()
        })

df = pd.DataFrame(rows)

# === TEMPORAL SPLIT ===
# All 4 cameras at same timestamp → same split (no data leakage)
unique_ts = sorted(df["timestamp"].unique())
ts_train, ts_temp = train_test_split(unique_ts, test_size=0.30, random_state=42)
ts_val, ts_test = train_test_split(ts_temp, test_size=0.50, random_state=42)

ts_train_set, ts_val_set, ts_test_set = set(ts_train), set(ts_val), set(ts_test)
df["split"] = df["timestamp"].apply(
    lambda t: "train" if t in ts_train_set else ("val" if t in ts_val_set else "test")
)

# === SAVE ===
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

# === REPORT ===
print(f"Total samples: {len(df)}")
print(f"Unique timestamps: {len(unique_ts)}")
print(f"\nSplit distribution:\n{df['split'].value_counts()}")
print(f"\nPer camera per split:\n{df.groupby('split')['camera'].value_counts().unstack()}")
print(f"\nSaved to {OUTPUT}")