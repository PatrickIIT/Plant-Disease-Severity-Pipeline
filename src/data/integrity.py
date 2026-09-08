# src/data/integrity.py
"""
Dataset integrity validation for the Leaf Lesion Severity pipeline.

Checks:
- split existence and image counts
- corrupt images
- dimension distribution
- empty labels
- healthy vs diseased image counts
- class instance counts
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image


def leaf_or_lesion(
    cls_name: str,
    leaf_keywords: List[str],
    lesion_keywords: List[str],
) -> str:
    """Map a class name to 'leaf', 'lesion', or 'other'."""
    n = str(cls_name).lower()
    if any(k.lower() in n for k in leaf_keywords):
        return "leaf"
    if any(k.lower() in n for k in lesion_keywords):
        return "lesion"
    return "other"


def validate_split(
    dataset_root: Path,
    split: str,
    class_names: List[str],
    leaf_keywords: List[str],
    lesion_keywords: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Validate one split (train / valid / test).

    Returns a report dictionary or None if the split does not exist.
    """
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"

    if not img_dir.exists():
        print(f"[skip] split '{split}' not present")
        return None

    img_files = sorted(img_dir.glob("*.*"))
    n_images = len(img_files)

    # ---- image dimensions & corruption ----
    dims: List[Tuple[int, int]] = []
    corrupt: List[Tuple[str, str]] = []

    for p in img_files:
        try:
            with Image.open(p) as im:
                dims.append(im.size)  # (w, h)
        except Exception as e:
            corrupt.append((p.name, str(e)))

    dim_counter = Counter(dims)

    # ---- label statistics ----
    class_instance_counts: Counter = Counter()
    healthy_leaf_images = 0
    diseased_leaf_images = 0
    empty_labels = 0

    for lp in lbl_dir.glob("*.txt"):
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        if not lines:
            empty_labels += 1
            healthy_leaf_images += 1
            continue

        cls_ids = [int(l.split()[0]) for l in lines]
        has_lesion = False

        for cid in cls_ids:
            if cid < 0 or cid >= len(class_names):
                continue
            cname = class_names[cid]
            class_instance_counts[cname] += 1
            if leaf_or_lesion(cname, leaf_keywords, lesion_keywords) == "lesion":
                has_lesion = True

        if has_lesion:
            diseased_leaf_images += 1
        else:
            healthy_leaf_images += 1

    return {
        "n_images": n_images,
        "n_corrupt_images": len(corrupt),
        "corrupt_files": corrupt,
        "unique_dims": dict(dim_counter),
        "n_empty_label_files": empty_labels,
        "class_instance_counts": dict(class_instance_counts),
        "healthy_leaf_images": healthy_leaf_images,
        "diseased_leaf_images": diseased_leaf_images,
    }


def run_integrity_check(
    dataset_root: str | Path,
    class_names: List[str],
    leaf_keywords: List[str] | None = None,
    lesion_keywords: List[str] | None = None,
    splits: List[str] | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Full integrity validation across the requested splits.

    Returns
    -------
    report : dict
        Per-split detailed statistics.
    split_df : pd.DataFrame
        Summary table (n_images, healthy, diseased, pct_diseased).
    """
    dataset_root = Path(dataset_root)
    leaf_keywords = leaf_keywords or ["0", "leaf"]
    lesion_keywords = lesion_keywords or ["1", "spot", "lesion"]
    splits = splits or ["train", "valid", "test"]

    print("Classes:", class_names)

    report: Dict[str, Any] = {}
    split_rows: List[Dict[str, Any]] = []

    for split in splits:
        split_report = validate_split(
            dataset_root=dataset_root,
            split=split,
            class_names=class_names,
            leaf_keywords=leaf_keywords,
            lesion_keywords=lesion_keywords,
        )
        if split_report is None:
            continue

        report[split] = split_report
        n_images = split_report["n_images"]
        healthy = split_report["healthy_leaf_images"]
        diseased = split_report["diseased_leaf_images"]

        split_rows.append(
            {
                "split": split,
                "n_images": n_images,
                "healthy_images": healthy,
                "diseased_images": diseased,
                "pct_diseased": round(100 * diseased / max(n_images, 1), 2),
            }
        )

    split_df = pd.DataFrame(split_rows)
    print(split_df.to_string(index=False))

    # ---- split-balance sanity check ----
    total = split_df["n_images"].sum()
    if total > 0:
        for _, row in split_df.iterrows():
            pct = 100 * row["n_images"] / total
            print(f"{row['split']:>6}: {pct:5.1f}% of total images")
            if row["split"] == "train" and pct < 60:
                print(
                    "  -> WARNING: train split is smaller than the conventional "
                    "~70% — check the export config."
                )

    return report, split_df


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Dataset integrity check")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_yaml_path = Path(args.dataset) / "data.yaml"
    with open(data_yaml_path) as f:
        data_yaml = yaml.safe_load(f)

    report, split_df = run_integrity_check(
        dataset_root=args.dataset,
        class_names=data_yaml["names"],
        leaf_keywords=cfg.get("classes", {}).get("leaf", ["0"]),
        lesion_keywords=cfg.get("classes", {}).get("lesion", ["1"]),
    )
    print("\nDetailed report keys:", list(report.keys()))
