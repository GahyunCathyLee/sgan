#!/usr/bin/env bash
set +e

MMAP="/home/gahyun/neighformer/data/highD/v4_lit_g2_slot3"
SPLIT="/home/gahyun/neighformer/data/highD/splits"
CKPT_DIR="ckpts"

COMMON="--use_highd 1 \
        --highd_mmap_path ${MMAP} \
        --highd_split_dir ${SPLIT} \
        --num_samples 20 \
        --dset_type test"

FEATURE_MODES=(baseline dimI)

for mode in "${FEATURE_MODES[@]}"; do
  for trial in 1 2 3 4 5; do
    python -m scripts.evaluate_model ${COMMON} \
      --model_path "${CKPT_DIR}/v4lit_${mode}_t${trial}_best.pt"
  done
done
