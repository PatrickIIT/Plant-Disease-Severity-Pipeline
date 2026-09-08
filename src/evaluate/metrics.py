# src/evaluate/metrics.py
"""
Evaluation metrics for the Leaf Lesion Severity pipeline.

- Mask mAP@50 / mAP50-95 via Ultralytics
- Severity MAE against ground-truth polygon ratios
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from ultralytics import YOLO

from src.models.severity import classify_name, compute_severity


def run_mask_map(
    model: YOLO,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = 512,
) -> Dict[str, Any]:
    """
    Run Ultralytics validation and return key mask / box metrics.
    """
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
    )

    summary = {
        "box_mAP50": float(metrics.box.map50),
        "mask_mAP50": float(metrics.seg.map50),
        "mask_mAP50_95": float(metrics.seg.map),
        "per_class_mask_mAP50_95": dict(
            zip(model.names.values(), metrics.seg.maps)
        ),
    }

    print("Box  mAP50 :", summary["box_mAP50"])
    print("Mask mAP50 :", summary["mask_mAP50"])
    print("Mask mAP50-95:", summary["mask_mAP50_95"])
    print("Per-class mask mAP50-95:", summary["per_class_mask_mAP50_95"])

    return summary


def gt_severity_from_label(
    label_path: str | Path,
    img_w: int,
    img_h: int,
    class_names: List[str],
    leaf_keywords: List[str],
    lesion_keywords: List[str],
) -> float:
    """
    Compute ground-truth severity % from a YOLO-seg label file.
    """
    leaf_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    lesion_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    lines = [l for l in Path(label_path).read_text().splitlines() if l.strip()]
    for line in lines:
        parts = line.split()
        cid = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        pts = pts.astype(np.int32)

        kind = classify_name(
            class_names[cid], leaf_keywords, lesion_keywords
        )
        target = (
            leaf_mask
            if kind == "leaf"
            else (lesion_mask if kind == "lesion" else None)
        )
        if target is not None:
            cv2.fillPoly(target, [pts], 1)

    total_leaf = (
        leaf_mask.sum()
        if leaf_mask.sum() > 0
        else (leaf_mask | lesion_mask).sum()
    )
    lesion_area = lesion_mask.sum()
    return float(100.0 * lesion_area / total_leaf) if total_leaf > 0 else 0.0


def severity_mae(
    model: YOLO,
    dataset_root: str | Path,
    class_names: List[str],
    leaf_keywords: List[str] | None = None,
    lesion_keywords: List[str] | None = None,
    split: str = "test",
    conf: float = 0.25,
    imgsz: int = 512,
    severity_scale: str = "chiang",
) -> Tuple[pd.DataFrame, float]:
    """
    Compare predicted severity vs GT polygon severity on a split.

    Returns
    -------
    severity_df : DataFrame with per-image gt / pred / abs_error
    mae : mean absolute error in percentage points
    """
    leaf_keywords = leaf_keywords or ["0", "leaf"]
    lesion_keywords = lesion_keywords or ["1", "spot", "lesion"]

    dataset_root = Path(dataset_root)
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"

    rows: List[Dict[str, Any]] = []

    for img_path in sorted(img_dir.glob("*.*")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        with PILImage.open(img_path) as im:
            w, h = im.size

        gt_pct = gt_severity_from_label(
            lbl_path, w, h, class_names, leaf_keywords, lesion_keywords
        )
        pred = compute_severity(
            img_path,
            model=model,
            leaf_keywords=leaf_keywords,
            lesion_keywords=lesion_keywords,
            conf=conf,
            imgsz=imgsz,
            severity_scale=severity_scale,
        )

        rows.append(
            {
                "image": img_path.name,
                "gt_severity_pct": gt_pct,
                "pred_severity_pct": pred["severity_pct"],
            }
        )

    severity_df = pd.DataFrame(rows)
    severity_df["abs_error"] = (
        severity_df["gt_severity_pct"] - severity_df["pred_severity_pct"]
    ).abs()
    mae = float(severity_df["abs_error"].mean()) if len(severity_df) else 0.0

    print(f"Severity MAE on {split} set: {mae:.3f} percentage points (n={len(severity_df)})")
    return severity_df, mae
