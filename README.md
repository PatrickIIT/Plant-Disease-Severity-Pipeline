# 🌿 Leaf Lesion Segmentation → Disease Severity Scoring Pipeline

End-to-end, reproducible pipeline for **instance segmentation of leaf lesions** and **quantitative disease severity estimation**, with an optional Vision-Language Model (VLM) verification layer.

**Dataset:** [leaf-lesion-segmentation-qfbaa](https://universe.roboflow.com/balajicv/leaf-lesion-segmentation-qfbaa) (Roboflow Universe, CC BY 4.0)

---

## ✨ Features

| Stage | Description |
|-------|-------------|
| **Task 1** | Automated data ingestion via Roboflow API + integrity validation |
| **Task 2** | YOLOv8-seg fine-tuning → pixel-ratio severity scoring → Qwen2.5-VL visual critique |
| **Task 3** | Mask mAP@50, severity MAE, DL-vs-VLM cross-verification matrix, failure-mode analysis |
| **Task 4** | ONNX export + FastAPI + Docker serving design |

**Severity formula**

\[
\text{Severity \%} = \frac{\text{Lesion Pixels}}{\text{Total Leaf Pixels}} \times 100
\]

---

## 📊 Results (Test Set)

| Metric | Value |
|--------|-------|
| Mask mAP@50 | *(fill after evaluation)* |
| Severity MAE | *(fill after evaluation)* |
| DL–VLM disagreement rate | *(fill after evaluation)* |

> Plots are available in the `results/` folder.

---

## 🗂️ Repository Structure

```text
leaf-lesion-severity-pipeline/
├── notebooks/                  # Full Colab / Kaggle notebook
├── src/                        # Modular Python source
│   ├── data/                   # Download & validation
│   ├── models/                 # Training, severity, VLM
│   ├── evaluate/               # Metrics & diagnostics
│   └── serve/                  # Export helpers
├── serving/                    # Production FastAPI + Docker
├── results/                    # Generated figures & tables
├── config/                     # Configuration
├── docs/                       # Architecture & literature notes
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/leaf-lesion-severity-pipeline.git
cd leaf-lesion-severity-pipeline
pip install -r requirements.txt
```

### 2. Set your Roboflow API key

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

Or edit `config/config.yaml`.

### 3. Run the full pipeline (notebook)

Open `notebooks/leaf_lesion_severity_pipeline.ipynb` in **Google Colab** or **Kaggle** (GPU runtime recommended).

### 4. Inference only (skip training)

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")          # download from Releases
results = model.predict("leaf.jpg")
```

---

## 📦 Model Weights

Trained weights (`best.pt`) are **not** stored in the repository (file size).

Download from:

- [GitHub Releases]([(https://github.com/PatrickIIT/Plant-Disease-Severity-Pipeline/)](https://github.com/PatrickIIT/Plant-Disease-Severity-Pipeline/))  
- or Kaggle / Hugging Face (link them here)

---

## 🐳 Serving with Docker

```bash
cd serving
docker build -t leaf-severity-api .
docker run --gpus all -p 8080:8080 leaf-severity-api
```

**API endpoint**

```bash
curl -X POST -F "file=@leaf.jpg" http://localhost:8080/score
```

Response example:

```json
{
  "dl": {
    "leaf_px": 245120,
    "lesion_px": 18340,
    "severity_pct": 7.482
  },
  "vlm": null
}
```

---

## 📚 Literature Backbone

The pipeline design is informed by:

- Dual-stage leaf → lesion segmentation (Shivaraj & Haladappa, AgriMamba)
- Background / shadow removal for field robustness (Astani & Hasheminejad – BaSPaC)
- Severity scale considerations (Chiang & Hong)
- Weakly-supervised and multi-task severity estimation literature (Rezaei et al., AgriFusionNet, RSD-YOLO)

See `docs/papers.md` for short summaries.

---

## ⚙️ Configuration

Key options in `config/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `MODEL_ARCH` | `yolov8n-seg.pt` | Backbone |
| `EPOCHS` | 50 | Training epochs |
| `IMG_SIZE` | 512 | Input resolution |
| `SEVERITY_SCALE` | `chiang` | Severity mapping |
| `ENABLE_DUAL_STAGE_REFINEMENT` | `true` | Crop-then-refine |
| `VLM_LOAD_4BIT` | `true` | 4-bit quantization for Qwen2.5-VL |

---

## 📄 Citation

If you use this pipeline or the underlying dataset, please cite:

```bibtex
@misc{leaf-lesion-segmentation-qfbaa_dataset,
  title        = {leaf lesion segmentation Dataset},
  type         = {Open Source Dataset},
  author       = {balajiCV},
  howpublished = {\url{https://universe.roboflow.com/balajicv/leaf-lesion-segmentation-qfbaa}},
  journal      = {Roboflow Universe},
  publisher    = {Roboflow},
  year         = {2025},
  month        = {nov}
}
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

- Code: MIT License  
- Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
