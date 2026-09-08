# scripts/train.sh
#!/usr/bin/env bash
set -euo pipefail

# Train YOLOv8-seg using config/config.yaml
# Usage:  bash scripts/train.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/config.yaml}"
DATA_YAML="${DATA_YAML:-dataset/data.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

if [[ ! -f "$DATA_YAML" ]]; then
  echo "data.yaml not found: $DATA_YAML"
  echo "Run the download step first."
  exit 1
fi

echo ">>> Training with config=$CONFIG  data=$DATA_YAML"
python -m src.models.train --config "$CONFIG" --data "$DATA_YAML"

echo ">>> Training finished."
