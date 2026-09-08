# scripts/evaluate.sh
#!/usr/bin/env bash
set -euo pipefail

# Run mask mAP + severity MAE on the test split
# Usage:  bash scripts/evaluate.sh [path/to/best.pt]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/config.yaml}"
WEIGHTS="${1:-weights/best.pt}"
DATA_YAML="${DATA_YAML:-dataset/data.yaml}"
DATASET_ROOT="${DATASET_ROOT:-dataset}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Weights not found: $WEIGHTS"
  echo "Download best.pt or pass the path as the first argument."
  exit 1
fi

echo ">>> Evaluating  weights=$WEIGHTS"
python - <<PY
from pathlib import Path
import yaml
from ultralytics import YOLO
from src.evaluate.metrics import run_mask_map, severity_mae

with open("$CONFIG") as f:
    cfg = yaml.safe_load(f)

model = YOLO("$WEIGHTS")
data_yaml = Path("$DATA_YAML")
dataset_root = Path("$DATASET_ROOT")

# class names from data.yaml
with open(data_yaml) as f:
    data = yaml.safe_load(f)
names = data["names"]

leaf_kw = cfg.get("classes", {}).get("leaf", ["0", "leaf"])
lesion_kw = cfg.get("classes", {}).get("lesion", ["1", "spot", "lesion"])
imgsz = cfg.get("training", {}).get("img_size", 512)
scale = cfg.get("options", {}).get("severity_scale", "chiang")

print("--- Mask mAP ---")
run_mask_map(model, data_yaml, split="test", imgsz=imgsz)

print("--- Severity MAE ---")
severity_df, mae = severity_mae(
    model=model,
    dataset_root=dataset_root,
    class_names=names,
    leaf_keywords=leaf_kw,
    lesion_keywords=lesion_kw,
    split="test",
    imgsz=imgsz,
    severity_scale=scale,
)
out = Path("results")
out.mkdir(exist_ok=True)
severity_df.to_csv(out / "severity_mae.csv", index=False)
print(f"Saved results/severity_mae.csv  (MAE={mae:.3f})")
PY

echo ">>> Evaluation finished."
