# Architecture

End-to-end design of the **Leaf Lesion Segmentation → Disease Severity Scoring** pipeline.

---

## 1. High-level flow

```text
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Raw leaf    │────▶│  YOLOv8-seg     │────▶│  Pixel-ratio     │
│  image       │     │  (leaf + lesion)│     │  severity %      │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  Optional VLM    │
                                              │  (Qwen2.5-VL)    │
                                              │  visual critique │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  Cross-verify    │
                                              │  + JSON response │
                                              └──────────────────┘
```

**Severity formula**

\[
\text{Severity \%} = \frac{\text{Lesion pixels}}{\text{Total leaf pixels}} \times 100
\]

If no explicit leaf mask is predicted, the union of leaf ∪ lesion is used as the leaf proxy.

---

## 2. Pipeline stages

| Stage | Module | Responsibility |
|-------|--------|----------------|
| **1. Ingestion** | `src/data/download.py` | Roboflow API download (YOLOv8 polygon export) |
| **1. Validation** | `src/data/integrity.py` | Split balance, dimensions, healthy/diseased counts |
| **2. Training** | `src/models/train.py` | Fine-tune YOLOv8-seg |
| **2. Scoring** | `src/models/severity.py` | Mask → severity % + ordinal grade / bracket |
| **2. VLM** | `src/models/vlm_verify.py` | Qwen2.5-VL structured JSON critique |
| **3. Metrics** | `src/evaluate/metrics.py` | Mask mAP, severity MAE |
| **3. Cross-check** | `src/evaluate/cross_verify.py` | DL vs VLM disagreement matrix |
| **4. Export** | `src/serve/export.py` | ONNX (and optional TensorRT) |
| **4. Serving** | `serving/app.py` | FastAPI + Docker |

---

## 3. Dual-stage refinement (optional)

When `ENABLE_DUAL_STAGE_REFINEMENT=true`:

1. **Stage A** – full-image segmentation → coarse leaf mask  
2. **Stage B** – crop to leaf bounding box (with padding) → re-segment lesions at higher relative resolution  

This follows the leaf-then-lesion pattern used in Shivaraj & Haladappa and AgriMamba-style pipelines.

---

## 4. Severity scales

| Scale | Behaviour |
|-------|-----------|
| `linear` | Equal 10 % bins |
| `horsfall_barratt` | Classic log-style intervals (narrow near 0/100 %) |
| `chiang` | Fine bins below 10 %, then 10 % steps (default) |

Coarse **visual brackets** used for VLM comparison:

- `healthy(0-2%)`
- `low(2-10%)`
- `moderate(10-25%)`
- `high(25-50%)`
- `severe(>50%)`

---

## 5. Cross-verification logic

```text
DL severity %  ──▶  severity_bracket
                         │
VLM JSON       ──▶  visual_severity_bracket + confidence
                         │
                         ▼
              bracket_distance + confidence threshold
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
      agree          borderline       disagree
   (dist = 0)       (dist = 1)      (dist ≥ 2)
```

Low VLM confidence → status `uncertain` (does not force a hard disagreement).

---

## 6. Serving architecture

```text
Client / Edge device
        │  HTTPS multipart image
        ▼
API Gateway / Load balancer
        │
        ▼
┌───────────────────────────────────────┐
│  FastAPI scoring service (Docker+GPU) │
│  1. Decode image                      │
│  2. YOLOv8-seg (ONNX Runtime / TRT)   │
│  3. Pixel-ratio severity              │
│  4. (async) VLM microservice          │
│  5. Merge → JSON response             │
└───────────────┬───────────────────────┘
                │
     ┌──────────┴──────────┐
     ▼                     ▼
 Segmentation           VLM verifier
 (ONNX / TensorRT)      (Qwen2.5-VL 4-bit)
```

**Deployment notes**

- **Edge** – ONNX Runtime (CPU/NPU); skip on-device VLM; upload uncertain frames to cloud.
- **Server** – TensorRT for lowest latency; VLM in a separate container so it never blocks the fast path.
- **Orchestration** – Docker Compose or Triton Inference Server for independent scaling.

---

## 7. Key artifacts

| Artifact | Location |
|----------|----------|
| Best weights | `weights/best.pt` (see Releases) |
| ONNX export | `export/leaf_lesion_seg.onnx` |
| FastAPI app | `serving/app.py` |
| Docker | `serving/Dockerfile`, `serving/docker-compose.yml` |
| Config | `config/config.yaml` |

---

## 8. Design principles

1. **Isolate then quantify** – leaf mask first, lesion mask second, area ratio last.  
2. **Deterministic core** – severity % is a pure pixel computation; VLM is advisory only.  
3. **Fail open on VLM** – if the VLM is unavailable or low-confidence, still return the DL measurement.  
4. **Reproducible config** – all hyperparameters live in `config/config.yaml`.  
5. **Production-ready packaging** – ONNX + FastAPI + Docker from day one.
