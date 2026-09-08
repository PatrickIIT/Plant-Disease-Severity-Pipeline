# serving/app.py
"""
Production FastAPI service for leaf-lesion severity scoring.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8080

Run in Docker:
    docker compose up --build
"""

from __future__ import annotations

import io
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "leaf_lesion_seg.onnx"   # place the exported ONNX next to this file
# MODEL_PATH = "leaf_lesion_seg.pt"   # or use the PyTorch weights

LEAF_KEYWORDS = ["0", "leaf"]
LESION_KEYWORDS = ["1", "spot", "lesion"]
VLM_ENABLED = False                  # flip when a VLM microservice is available

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
model = YOLO(MODEL_PATH, task="segment")
NAMES = model.names

app = FastAPI(
    title="Leaf Lesion Severity Scoring API",
    version="1.0.0",
    description="YOLOv8-seg → pixel-ratio disease severity",
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
    return {"status": "ok", "model": MODEL_PATH, "vlm_enabled": VLM_ENABLED}


@app.post("/score")
async def score(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dl_result = compute_severity(img_bgr)

    response = {"dl": dl_result, "vlm": None}

    # Optional: call external VLM microservice when VLM_ENABLED is True
    if VLM_ENABLED:
        pass

    return JSONResponse(response)
