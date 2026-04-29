#!/usr/bin/env bash
set +e

MMAP="/home/gahyun/neighformer/data/highD/v4_lit_g2_slot3"
SPLIT="/home/gahyun/neighformer/data/highD/splits"

COMMON="--use_highd 1 \
        --highd_mmap_path ${MMAP} \
        --highd_split_dir ${SPLIT} \
        --pooling_type highd_pool \
        --batch_size 1024 \
        --num_epochs 100 \
        --g_learning_rate 1e-4 \
        --d_learning_rate 1e-4 \
        --g_steps 1 \
        --d_steps 2 \
        --clipping_threshold_g 1.5 \
        --best_k 6 \
        --print_every 50 \
        --checkpoint_every 100 \
        --restore_from_checkpoint 0 \
        --output_dir ckpts"

SEEDS=(42 1234 3407 0 777)

# baseline: ego+nb(6D)
for i in "${!SEEDS[@]}"; do
  trial=$((i + 1))
  python -m scripts.train ${COMMON} \
    --feature_mode baseline \
    --seed "${SEEDS[$i]}" \
    --checkpoint_name "v4lit_baseline_t${trial}"
done

# dimI: ego+nb(6D)+dim(8)+I(9)
for i in "${!SEEDS[@]}"; do
  trial=$((i + 1))
  python -m scripts.train ${COMMON} \
    --feature_mode dimI \
    --seed "${SEEDS[$i]}" \
    --checkpoint_name "v4lit_dimI_t${trial}"
done
