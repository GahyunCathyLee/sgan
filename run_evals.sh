#!/bin/bash

# 공통 옵션
COMMON_T="--use_highd 1 \
--highd_mmap_path data/highD/dimI \
--highd_split_dir data/highD/splits \
--dset_type test \
--scenario_labels data/highD/dimI/scenario_labels.csv"

# 모델 경로
MODEL_PATH="ckpts/highD0-1_best.pt"

# 실행
python -m scripts.evaluate_model ${COMMON_T} --model_path ${MODEL_PATH}
