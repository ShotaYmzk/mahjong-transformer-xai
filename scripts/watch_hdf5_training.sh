#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_PATH="outputs/impl1/hdf5_watchdog.log"
METRICS_PATH="outputs/impl1/hdf5_10epoch_metrics.csv"
TRAIN_LOG_PATH="outputs/impl1/hdf5_10epoch.jsonl"
CHECKPOINT_PATH="outputs/impl1/hdf5_10epoch.pt"
TARGET_EPOCHS=10

mkdir -p outputs/impl1

timestamp() {
  date -Is
}

is_complete() {
  python - <<'PY'
import csv
from pathlib import Path
path = Path("outputs/impl1/hdf5_10epoch_metrics.csv")
if not path.exists():
    raise SystemExit(1)
rows = list(csv.DictReader(path.open()))
if not rows:
    raise SystemExit(1)
last_epoch = int(float(rows[-1]["epoch"]))
raise SystemExit(0 if last_epoch >= 10 else 1)
PY
}

while true; do
  if is_complete; then
    echo "$(timestamp) training already complete" | tee -a "$LOG_PATH"
    exit 0
  fi

  if pgrep -f "[t]rain_transformer_v2_hdf5.py --shards-dir outputs/impl1/hdf5_shards_250k" >/dev/null; then
    echo "$(timestamp) training is running" | tee -a "$LOG_PATH"
    sleep 300
    continue
  fi

  echo "$(timestamp) training not running; starting/resuming" | tee -a "$LOG_PATH"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/train_transformer_v2_hdf5.py \
    --shards-dir outputs/impl1/hdf5_shards_250k \
    --output "$CHECKPOINT_PATH" \
    --epochs "$TARGET_EPOCHS" \
    --batch-size 4096 \
    --valid-batch-size 1024 \
    --d-model 64 \
    --n-layers 2 \
    --n-heads 4 \
    --d-ff 256 \
    --valid-shards 8 \
    --log "$TRAIN_LOG_PATH" \
    --metrics-csv "$METRICS_PATH" \
    --resume 2>&1 | tee -a "$LOG_PATH"

  echo "$(timestamp) training process exited; watchdog will re-check" | tee -a "$LOG_PATH"
  sleep 30
done
