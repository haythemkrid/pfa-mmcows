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
        lbl_candidates = [
            dataset_root / "labels" / date_folder / cam,
            dataset_root / "labels" / "combined" / date_folder / cam,
        ]
        lbl_dir = next((path for path in lbl_candidates if path.exists()), lbl_candidates[0])

        if not img_dir.exists():
            print(f"[WARN] Missing image dir: {img_dir}")
            continue
        if not lbl_dir.exists():
            print(f"[WARN] Missing label dir: {lbl_dir}")
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
                    "img_path": img_path.absolute(),
                    "lbl_path": lbl_path.absolute(),
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
