# src/serve/export.py
"""
Model export helpers (ONNX / TensorRT) for the Leaf Lesion Severity pipeline.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def export_onnx(
    weights: str | Path,
    export_dir: str | Path = "export",
    dynamic: bool = True,
    simplify: bool = True,
    opset: int = 17,
    filename: str = "leaf_lesion_seg.onnx",
) -> Path:
    """
    Export a trained YOLO segmentation model to ONNX and copy it
    into the export directory.

    Returns
    -------
    Path
        Path to the copied ONNX file inside export_dir.
    """
    weights = Path(weights)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    onnx_path = model.export(
        format="onnx",
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
    )
    onnx_path = Path(onnx_path)

    dest = export_dir / filename
    shutil.copy(onnx_path, dest)

    # Also keep a copy of the original .pt next to it
    pt_dest = export_dir / "leaf_lesion_seg.pt"
    shutil.copy(weights, pt_dest)

    print("ONNX export ->", dest)
    print("Weights copy ->", pt_dest)
    return dest


def export_tensorrt(
    weights: str | Path,
    half: bool = True,
    workspace: int = 4,
) -> Optional[Path]:
    """
    Export to TensorRT engine (only works on a machine with TensorRT installed).

    Returns the engine path, or None if export is skipped / fails.
    """
    try:
        model = YOLO(str(weights))
        engine_path = model.export(
            format="engine",
            dynamic=True,
            half=half,
            workspace=workspace,
        )
        print("TensorRT engine ->", engine_path)
        return Path(engine_path)
    except Exception as e:
        print(f"TensorRT export skipped / failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLO-seg model")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--export-dir", default="export")
    parser.add_argument("--tensorrt", action="store_true")
    args = parser.parse_args()

    export_onnx(args.weights, export_dir=args.export_dir)
    if args.tensorrt:
        export_tensorrt(args.weights)
