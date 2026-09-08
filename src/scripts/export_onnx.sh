# scripts/export_onnx.sh
#!/usr/bin/env bash
set -euo pipefail

# Export best.pt → ONNX and copy into export/ (+ optional serving/)
# Usage:  bash scripts/export_onnx.sh [path/to/best.pt]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WEIGHTS="${1:-weights/best.pt}"
EXPORT_DIR="${EXPORT_DIR:-export}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Weights not found: $WEIGHTS"
  exit 1
fi

echo ">>> Exporting ONNX from $WEIGHTS"
python -m src.serve.export --weights "$WEIGHTS" --export-dir "$EXPORT_DIR"

# Also copy into serving/ so Docker can pick it up
if [[ -d serving ]]; then
  cp -f "$EXPORT_DIR/leaf_lesion_seg.onnx" serving/leaf_lesion_seg.onnx
  echo "Copied ONNX → serving/leaf_lesion_seg.onnx"
fi

echo ">>> Export finished."
