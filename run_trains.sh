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

# ── c0: ego+nb(6D, no importance)  |  5 trials ──────────────────────────────
python -m scripts.train ${COMMON} --use_Iy 0 --use_I 0 --seed 42   --checkpoint_name v4lit_c0_t1
python -m scripts.train${COMMON} --use_Iy 0 --use_I 0 --seed 1234 --checkpoint_name v4lit_c0_t2
python -m scripts.train ${COMMON} --use_Iy 0 --use_I 0 --seed 3407 --checkpoint_name v4lit_c0_t3
python -m scripts.train ${COMMON} --use_Iy 0 --use_I 0 --seed 0    --checkpoint_name v4lit_c0_t4
python -m scripts.train ${COMMON} --use_Iy 0 --use_I 0 --seed 777  --checkpoint_name v4lit_c0_t5

# ── c2: ego+nb(6D)+I_y  |  5 trials ─────────────────────────────────────────
python -m scripts.train ${COMMON} --use_Iy 1 --seed 42   --checkpoint_name v4lit_c2_t1
python -m scripts.train ${COMMON} --use_Iy 1 --seed 1234 --checkpoint_name v4lit_c2_t2
python -m scripts.train ${COMMON} --use_Iy 1 --seed 3407 --checkpoint_name v4lit_c2_t3
python -m scripts.train ${COMMON} --use_Iy 1 --seed 0    --checkpoint_name v4lit_c2_t4
python -m scripts.train ${COMMON} --use_Iy 1 --seed 777  --checkpoint_name v4lit_c2_t5
