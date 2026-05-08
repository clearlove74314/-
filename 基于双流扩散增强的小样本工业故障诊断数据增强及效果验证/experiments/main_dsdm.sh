#!/bin/bash

echo "=== Preprocessing Data ==="
python src/data/preprocess.py \
    --raw_dir data/raw \
    --save_dir data/processed \
    --fs 12000 \
    --window_size 2048 \
    --stride 1024 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --gap 2048

echo "=== Checking for Data Leakage ==="
python src/data/leakage_check.py \
    --index_dir data/processed/index_record

echo "=== Calculating Fault Frequencies ==="
python src/data/frequency_analysis.py \
    --fs 12000 \
    --rpm 1500 \
    --num_balls 8 \
    --diameter_ratio 0.206 \
    --contact_angle 0 \
    --gear_teeth 32

echo "=== Building Few-shot Tasks ==="
python src/data/build_fewshot_task.py \
    --train_dir data/processed/train \
    --save_dir data/processed/support_set \
    --num_classes 5 \
    --shot 5 \
    --seeds 2024 2025 2026 2027 2028

echo "=== Pretraining DSDM ==="
python src/train/pretrain_dsdm.py \
    --config config/dsdm.yaml \
    --train_dir data/processed/train \
    --val_dir data/processed/val \
    --save_dir outputs/checkpoints/dsdm_pretrain \
    --seed 2024