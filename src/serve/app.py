# src/serve/app.py
"""
FastAPI service for DL severity scoring (+ optional VLM verification).

Run locally:
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8080

Or from the serving/ directory after copying this file:
    uvicorn app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration (override via environment variables in production)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "export" / "leaf_lesion_seg.onnx"
# Fallback to .pt if ONNX is not present
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "export" / "leaf_lesion_seg.pt"

LEAF_KEYWORDS = ["0", "leaf"]
LESION_KEYWORDS = ["1", "spot", "lesion"]
VLM_ENABLED = False          # set True once a VLM service is wired in
VLM_CONFIDENCE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
model = YOLO(str(MODEL_PATH), task="segment")
NAMES = model.names

app = FastAPI(
    title="Leaf Lesion Severity Scoring API",
    version="1.0.0",
    description="YOLOv8-seg → pixel-ratio severity (+ optional VLM critique)",
)


def classify_name(name: str) -> str:
    n = str(name).lower()
    if any(k in n for k in LEAF_KEYWORDS):
        return "leaf"
    if any(k in n for k in LESION_KEYWORDS):
        return "lesion"
    return "other"


def severity_bracket(severity_pct: float) -> str:
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


def compute_severity(image: np.ndarray, conf: float = 0.25) -> Dict[str, Any]:
    """Run segmentation and return leaf/lesion pixel counts + severity %."""
    h, w = image.shape[:2]
    result = model.predict(image, conf=conf, verbose=False)[0]

    leaf_mask = np.zeros((h, w), dtype=bool)
    lesion_mask = np.zeros((h, w), dtype=bool)

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for m, cid in zip(masks, cls_ids):
            m_r = cv2.resize(
                m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            kind = classify_name(NAMES[cid])
            if kind == "leaf":
                leaf_mask |= m_r
            elif kind == "lesion":
                lesion_mask |= m_r

    total_leaf = int(leaf_mask.sum()) if leaf_mask.any() else int((leaf_mask | lesion_mask).sum())
    lesion_area = int(lesion_mask.sum())
    severity_pct = float(100.0 * lesion_area / total_leaf) if total_leaf > 0 else 0.0

    return {
        "leaf_px": total_leaf,
        "lesion_px": lesion_area,
        "severity_pct": round(severity_pct, 3),
        "severity_bracket": severity_bracket(severity_pct),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": str(MODEL_PATH.name),
        "vlm_enabled": VLM_ENABLED,
    }


@app.post("/score")
async def score(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Upload a JPEG or PNG image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image file.")

    try:
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dl_result = compute_severity(img_bgr)

    response: Dict[str, Any] = {
        "dl": dl_result,
        "vlm": None,
    }

    # Optional: call an external VLM microservice here when VLM_ENABLED is True
    if VLM_ENABLED:
        # Example:
        # import requests, base64
        # b64 = base64.b64encode(raw).decode()
        # vlm_resp = requests.post(VLM_SERVICE_URL, json={
        #     "image_b64": b64,
        #     "dl_severity_pct": dl_result["severity_pct"],
        # }).json()
        # response["vlm"] = vlm_resp
        pass

    return JSONResponse(response)
