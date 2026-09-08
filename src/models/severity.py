# src/models/severity.py
"""
Pixel-ratio severity scoring from leaf / lesion segmentation masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


def classify_name(
    name: str,
    leaf_keywords: List[str],
    lesion_keywords: List[str],
) -> str:
    n = str(name).lower()
    if any(k.lower() in n for k in leaf_keywords):
        return "leaf"
    if any(k.lower() in n for k in lesion_keywords):
        return "lesion"
    return "other"


def bucket_severity(
    severity_pct: float,
    scale: str = "chiang",
) -> Dict[str, Any]:
    """
    Map continuous severity % to an ordinal grade.

    scale options:
      - linear
      - horsfall_barratt
      - chiang  (fine bins below 10 %, then 10 % steps)
    """
    s = max(0.0, float(severity_pct))

    if scale == "horsfall_barratt":
        # Classic HB mid-points (simplified)
        bins = [0, 3, 6, 12, 25, 50, 75, 88, 94, 97, 100]
        labels = ["0", "0-3", "3-6", "6-12", "12-25", "25-50",
                  "50-75", "75-88", "88-94", "94-97", "97-100"]
    elif scale == "chiang":
        bins = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = ["0", "0-1", "1-2", "2-5", "5-10", "10-20", "20-30",
                  "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    else:  # linear 10 % bins
        bins = list(range(0, 110, 10))
        labels = [f"{i}-{i+10}" for i in range(0, 100, 10)] + ["100"]

    for i in range(len(bins) - 1):
        if bins[i] <= s < bins[i + 1]:
            return {
                "grade_index": i,
                "grade_label": labels[i],
                "severity_pct": round(s, 3),
            }
    return {
        "grade_index": len(labels) - 1,
        "grade_label": labels[-1],
        "severity_pct": round(s, 3),
    }


def severity_bracket(severity_pct: float) -> str:
    """Coarse visual brackets used by the VLM cross-check."""
    s = float(severity_pct)
    if s < 2:
        return "healthy(0-2%)"
    if s < 10:
        return "low(2-10%)"
    if s < 25:
        return "moderate(10-25%)"
    if s < 50:
        return "high(25-50%)"
    return "severe(>50%)"


def compute_severity(
    image: Union[str, Path, np.ndarray],
    model: YOLO,
    leaf_keywords: List[str] | None = None,
    lesion_keywords: List[str] | None = None,
    conf: float = 0.25,
    imgsz: int = 512,
    severity_scale: str = "chiang",
) -> Dict[str, Any]:
    """
    Run segmentation and compute Severity % = lesion_px / leaf_px * 100.

    Parameters
    ----------
    image : path or BGR numpy array
    model : loaded YOLO segmentation model
    """
    leaf_keywords = leaf_keywords or ["0", "leaf"]
    lesion_keywords = lesion_keywords or ["1", "spot", "lesion"]

    result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
    h, w = result.orig_shape
    names = model.names

    leaf_mask = np.zeros((h, w), dtype=bool)
    lesion_mask = np.zeros((h, w), dtype=bool)

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for m, cid in zip(masks, cls_ids):
            m_resized = cv2.resize(
                m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            kind = classify_name(names[cid], leaf_keywords, lesion_keywords)
            if kind == "leaf":
                leaf_mask |= m_resized
            elif kind == "lesion":
                lesion_mask |= m_resized

    total_leaf = int(leaf_mask.sum()) if leaf_mask.any() else int((leaf_mask | lesion_mask).sum())
    lesion_area = int(lesion_mask.sum())
    severity_pct = float(100.0 * lesion_area / total_leaf) if total_leaf > 0 else 0.0

    grade = bucket_severity(severity_pct, scale=severity_scale)
    bracket = severity_bracket(severity_pct)

    return {
        "leaf_px": total_leaf,
        "lesion_px": lesion_area,
        "severity_pct": round(severity_pct, 3),
        "severity_grade": grade,
        "severity_bracket": bracket,
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask,
        "result": result,
    }
