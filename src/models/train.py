# src/models/train.py
"""
YOLOv8-seg training for the Leaf Lesion Severity pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO


def train_segmentation_model(
    data_yaml: str | Path,
    model_arch: str = "yolov8n-seg.pt",
    epochs: int = 50,
    imgsz: int = 512,
    batch: int = 32,
    project: str = "runs/severity_seg",
    name: str = "leaf_lesion_seg",
    patience: int = 15,
    device: int | str = 0,
    workers: int = 4,
    cache: bool = True,
    **augment_kwargs,
) -> str:
    """
    Fine-tune a YOLOv8 segmentation model.

    Returns
    -------
    str
        Path to the best weights (best.pt).
    """
    model = YOLO(model_arch)

    default_aug = dict(
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=15,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        erasing=0.2,
    )
    default_aug.update(augment_kwargs)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=patience,
        device=device,
        workers=workers,
        cache=cache,
        **default_aug,
    )

    best_weights = str(model.trainer.best)
    print("Best weights:", best_weights)
    return best_weights


def train_from_config(config: Dict[str, Any], data_yaml: str | Path) -> str:
    """Convenience wrapper that reads nested config.yaml structure."""
    t = config.get("training", {})
    return train_segmentation_model(
        data_yaml=data_yaml,
        model_arch=t.get("model_arch", "yolov8n-seg.pt"),
        epochs=t.get("epochs", 50),
        imgsz=t.get("img_size", 512),
        batch=t.get("batch", 32),
        project=t.get("project_dir", "runs/severity_seg"),
        name=t.get("run_name", "leaf_lesion_seg"),
        patience=t.get("patience", 15),
        device=t.get("device", 0),
    )


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train YOLOv8-seg")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default="dataset/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_from_config(cfg, args.data)
