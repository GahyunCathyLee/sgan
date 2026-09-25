#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gahyun/miniconda3/envs/tf/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

WARMUP="${WARMUP:-1000}"
ITERS="${ITERS:-10000}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs/latency}"
mkdir -p "$LOG_DIR"

cases=(
  "exiD-baseline|exiD|ckpts/exiD0-5_best.pt"
  "exiD-+I|exiD|ckpts/exiD2-5_best.pt"
  "highD-baseline|highD|ckpts/highD0-4_best.pt"
  "highD-+I|highD|ckpts/highD2-3_best.pt"
)

for row in "${cases[@]}"; do
  IFS='|' read -r name dataset ckpt <<< "$row"
  log_path="${LOG_DIR}/${name}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] ${name}: missing ${ckpt}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" -m scripts.evaluate_model
    --model_path "$ckpt"
    --dset_type test
    --measure_time
    --latency_warmup "$WARMUP"
    --latency_iters "$ITERS"
    --use_highd 1
    --highd_mmap_path "data/${dataset}/dimI"
    --highd_split_dir "data/${dataset}/splits"
  )

  echo "[RUN] ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
