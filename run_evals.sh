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

# ── c0: ego+nb(6D, no importance)  |  5 trials ──────────────────────────────
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c0_t1_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c0_t2_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c0_t3_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c0_t4_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c0_t5_best.pt

# ── c2: ego+nb(6D)+I_y  |  5 trials ─────────────────────────────────────────
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c2_t1_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c2_t2_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c2_t3_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c2_t4_best.pt
python -m scripts.evaluate_model ${COMMON} --model_path ${CKPT_DIR}/v4lit_c2_t5_best.pt
