"""Build a merged visual dataset index for YOLO workflows.

This script scans image/label folders, keeps samples with both files,
assigns train/val/test splits, and saves both pickle and CSV outputs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual dataset index.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("store/data/raw/visual_data"),
        help="Root directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--date-folder",
        type=str,
        default="0725",
        help="Date subfolder under images/ and labels/.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="cam_1,cam_2,cam_3,cam_4",
        help="Comma-separated camera folders.",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.70,0.15,0.15",
        help="Comma-separated train,val,test ratios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split shuffling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where dataset_index.pkl and dataset_index.csv are written.",
    )
    return parser.parse_args()


def build_dataframe(
    dataset_root: Path,
    date_folder: str,
    cameras: list[str],
    image_extensions: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cam in cameras:
        img_dir = dataset_root / "images" / date_folder / cam
        lbl_dir = dataset_root / "labels" / date_folder / cam

        if not img_dir.exists():
            print(f"[WARN] Missing image dir: {img_dir}")
            continue

        img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions)

        for img_path in img_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            rows.append(
                {
                    "sample_id": f"{cam}_{img_path.stem}",
                    "cam": cam,
                    "orig_stem": img_path.stem,
                    "img_path": img_path.resolve(),
                    "lbl_path": lbl_path.resolve(),
                    "img_ext": img_path.suffix.lower(),
                }
            )

    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.DataFrame:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    df = df.sort_values(["cam", "orig_stem"]).reset_index(drop=True)
    n = len(df)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n))

    rng = random.Random(seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    split_col = [""] * n
    for idx in train_idx:
        split_col[idx] = "train"
    for idx in val_idx:
        split_col[idx] = "val"
    for idx in test_idx:
        split_col[idx] = "test"

    df["split"] = split_col
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("-" * 60)
    print(f"Total samples: {len(df)}")
    print("Per camera:")
    print(df.groupby("cam")["sample_id"].count().to_string())
    print("Per split:")
    print(df.groupby("split")["sample_id"].count().to_string())
    print("Per camera x split:")
    print(df.groupby(["cam", "split"])["sample_id"].count().unstack(fill_value=0).to_string())
    print("-" * 60)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = [part.strip() for part in args.cameras.split(",") if part.strip()]
    ratios = tuple(float(part.strip()) for part in args.split_ratios.split(","))
    if len(ratios) != 3:
        raise ValueError("split-ratios must contain exactly 3 values: train,val,test")

    df = build_dataframe(
        dataset_root=dataset_root,
        date_folder=args.date_folder,
        cameras=cameras,
        image_extensions={".jpg", ".jpeg", ".png"},
    )

    if df.empty:
        raise RuntimeError("No valid samples found. Verify dataset root/date/camera folders.")

    df = assign_splits(df, ratios=ratios, seed=args.seed)
    print_summary(df)

    pkl_path = output_dir / "dataset_index.pkl"
    csv_path = output_dir / "dataset_index.csv"

    df.to_pickle(pkl_path)
    df.assign(img_path=df["img_path"].astype(str), lbl_path=df["lbl_path"].astype(str)).to_csv(
        csv_path,
        index=False,
    )

    print(f"[OK] Saved index pickle: {pkl_path}")
    print(f"[OK] Saved index csv: {csv_path}")


if __name__ == "__main__":
    main()
"""
A1 — Rebuild merged dataset index (DataFrame strategy)
=======================================================
No files are copied. Everything lives in two DataFrames:
  - df_images : one row per image file
  - df_labels : one row per label file, joined to df_images on sample_id

A single merged DataFrame `df` is saved as  dataset_index.pkl  and
dataset_index.csv (human-readable) for inspection and reuse in A2+.

Columns in df
-------------
sample_id   : str   cam_X_<orig_stem>          unique key across all cameras
cam         : str   cam_1 / cam_2 / cam_3 / cam_4
orig_stem   : str   original filename stem (no extension)
img_path    : Path  absolute path to image file
lbl_path    : Path  absolute path to label .txt file
img_ext     : str   .jpg / .png etc.
split       : str   train / val / test

Expected input layout
---------------------
dataset_root/
  images/0725/cam_1/  *.jpg
  labels/0725/cam_1/  *.txt
  ... (same for cam_2, cam_3, cam_4)
"""

import random
import pickle
from pathlib import Path

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\DELL\Desktop\test\mmcows\data\raw\visual_data")          # ← adjust to your actual root
DATE_FOLDER  = "0725"
CAMERAS      = ["cam_1", "cam_2", "cam_3", "cam_4"]
IMG_EXTS     = {".jpg", ".jpeg", ".png"}
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
RANDOM_SEED  = 42
# ─────────────────────────────────────────────────────────────────────────────


def build_dataframe(dataset_root: Path, date: str, cameras: list[str]) -> pd.DataFrame:
    """
    Walk each camera folder and build a raw index DataFrame.
    Only rows where both image AND label exist are kept.
    """
    rows = []
    for cam in cameras:
        img_dir = dataset_root / "images" / date / cam
        lbl_dir = dataset_root / "labels" / date / cam

        if not img_dir.exists():
            print(f"[WARN] Missing image dir: {img_dir}")
            continue

        img_files = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )

        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                print(f"[WARN] No label for {img_path.name} — skipped.")
                continue

            rows.append({
                "sample_id": f"{cam}_{img_path.stem}",
                "cam":       cam,
                "orig_stem": img_path.stem,
                "img_path":  img_path.resolve(),
                "lbl_path":  lbl_path.resolve(),
                "img_ext":   img_path.suffix.lower(),
            })

    df = pd.DataFrame(rows)
    return df


def assign_splits(df: pd.DataFrame, ratios: tuple, seed: int) -> pd.DataFrame:
    """
    Time-based split: sort by (cam, orig_stem) to preserve temporal order,
    slice into train/val/test, then shuffle within each split.
    Returns df with a new 'split' column.
    """
    df = df.sort_values(["cam", "orig_stem"]).reset_index(drop=True)

    n     = len(df)
    n_tr  = int(n * ratios[0])
    n_val = int(n * ratios[1])

    rng = random.Random(seed)

    train_idx = list(range(0, n_tr))
    val_idx   = list(range(n_tr, n_tr + n_val))
    test_idx  = list(range(n_tr + n_val, n))

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    split_col = [""] * n
    for i in train_idx: split_col[i] = "train"
    for i in val_idx:   split_col[i] = "val"
    for i in test_idx:  split_col[i] = "test"

    df["split"] = split_col
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'─'*50}")
    print(f"Total samples : {len(df)}")
    print(f"\nPer camera:")
    print(df.groupby("cam")["sample_id"].count().to_string())
    print(f"\nPer split:")
    print(df.groupby("split")["sample_id"].count().to_string())
    print(f"\nPer camera x split:")
    print(df.groupby(["cam", "split"])["sample_id"].count().unstack(fill_value=0).to_string())
    print(f"{'─'*50}\n")


def main() -> None:
    out_dir  = DATASET_ROOT
    pkl_path = out_dir / "dataset_index.pkl"
    csv_path = out_dir / "dataset_index.csv"
    print(DATASET_ROOT)
    print(DATE_FOLDER)
    print(CAMERAS)

    # ── Build ─────────────────────────────────────────────────────────────
    print("Building DataFrame index ...")
    df = build_dataframe(DATASET_ROOT, DATE_FOLDER, CAMERAS)

    if df.empty:
        print("[ERROR] No samples found. Check DATASET_ROOT and folder names.")
        return

    # ── Split ─────────────────────────────────────────────────────────────
    print("Assigning splits ...")
    df = assign_splits(df, SPLIT_RATIOS, RANDOM_SEED)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(df)

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_pickle(pkl_path)
    # CSV stores paths as strings (pickle preserves Path objects)
    df.assign(
        img_path=df["img_path"].astype(str),
        lbl_path=df["lbl_path"].astype(str),
    ).to_csv(csv_path, index=False)

    print(f"Saved:")
    print(f"  {pkl_path}  <- load in A2+ with pd.read_pickle()")
    print(f"  {csv_path}  <- human-readable inspection")
    print("\nA1 complete.")


if __name__ == "__main__":
    main()
