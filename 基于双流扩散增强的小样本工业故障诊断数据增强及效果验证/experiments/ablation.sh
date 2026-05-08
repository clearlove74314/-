#!/bin/bash

echo "=== Running Ablation Experiments ==="
python src/train/run_ablation.py \
    --config config/ablation.yaml \
    --train_dir data/processed/train \
    --val_dir data/processed/val \
    --test_dir data/processed/test \
    --support_root data/processed/support_set \
    --save_dir outputs/logs/ablation \
    --seeds 2024 2025 2026 2027 2028