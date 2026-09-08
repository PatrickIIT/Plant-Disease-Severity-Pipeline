# Model Weights

Trained YOLOv8-seg weights for the **Leaf Lesion Severity** pipeline are stored in this repository.

## Download

| File | Description | Size (approx.) | Link |
|------|-------------|----------------|------|
| `best.pt` | Best checkpoint from training (recommended) | ~6–7 MB | [GitHub Releases]([https://github.com/PatrickIIT/Plant-Disease-Severity-Pipeline/blob/main/src/weights/best%20(2).zip](https://github.com/PatrickIIT/Plant-Disease-Severity-Pipeline/blob/main/src/weights/best%20(2).zip)) |
| `last.pt` | Last epoch checkpoint | ~6–7 MB | same release |
| `leaf_lesion_seg.onnx` | ONNX export for serving | ~13 MB | same release |


## How to use

### Inference (Python)

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")          # or path to the downloaded file
results = model.predict("leaf.jpg")
```

### Severity scoring

```python
from src.models.severity import compute_severity

out = compute_severity("leaf.jpg", model=model)
print(out["severity_pct"], out["severity_bracket"])
```

### Serving

Place `leaf_lesion_seg.onnx` next to `serving/app.py`, then:

```bash
cd serving
docker compose up --build
```

## Training details (reference)

| Setting | Value |
|---------|-------|
| Architecture | `yolov8n-seg.pt` |
| Epochs | 50 |
| Image size | 512 |
| Batch size | 32 |
| Classes | `0` (leaf), `1` (lesion/spot) |
| Dataset | [leaf-lesion-segmentation-qfbaa](https://universe.roboflow.com/balajicv/leaf-lesion-segmentation-qfbaa) v1 |

## License

Weights inherit the dataset license (**CC BY 4.0**). Please cite the dataset when using them.
