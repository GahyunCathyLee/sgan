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

# Optional data overrides:
#   DATA_ROOT=/path/holding/data_dirs ./run_latency.sh
#   EXID_MMAP_DIR=/path/to/exiD/dimI EXID_SPLIT_DIR=/path/to/exiD/splits ./run_latency.sh
# Optional checkpoint overrides:
#   CKPT_ROOT=/path/to/sgan_ckpts ./run_latency.sh
#   EXID_BASE_CKPT=/path/to/exiD0-5_best.pt EXID_I_CKPT=/path/to/exiD2-5_best.pt ./run_latency.sh

cases=(
  "exiD-baseline|exiD|ckpts/exiD0-5_best.pt"
  "exiD-+I|exiD|ckpts/exiD2-5_best.pt"
)

for row in "${cases[@]}"; do
  IFS='|' read -r name dataset ckpt <<< "$row"
  condition="${name#*-}"
  log_path="${LOG_DIR}/${name}.log"

  ckpt_key=""
  if [[ "$condition" == "baseline" ]]; then
    ckpt_key="${EXID_BASE_CKPT:-}"
  else
    ckpt_key="${EXID_I_CKPT:-}"
  fi
  if [[ -n "$ckpt_key" ]]; then
    ckpt="$ckpt_key"
  elif [[ -n "${CKPT_ROOT:-}" ]]; then
    ckpt="${CKPT_ROOT}/${ckpt#ckpts/}"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] ${name}: missing ${ckpt}"
    continue
  fi

  mmap_dir="data/${dataset}/dimI"
  split_dir="data/${dataset}/splits"
  mmap_dir="${EXID_MMAP_DIR:-$mmap_dir}"
  split_dir="${EXID_SPLIT_DIR:-$split_dir}"
  if [[ -n "${DATA_ROOT:-}" ]]; then
    mmap_dir="${DATA_ROOT}/${dataset}/dimI"
    split_dir="${DATA_ROOT}/${dataset}/splits"
  fi

  cmd=(
    "$PYTHON_BIN" -m scripts.evaluate_model
    --model_path "$ckpt"
    --dset_type test
    --measure_time
    --latency_warmup "$WARMUP"
    --latency_iters "$ITERS"
    --use_highd 1
    --highd_mmap_path "$mmap_dir"
    --highd_split_dir "$split_dir"
  )

  echo "[RUN] ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
