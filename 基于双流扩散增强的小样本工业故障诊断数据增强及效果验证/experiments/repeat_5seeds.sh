#!/bin/bash

SEEDS=(2024 2025 2026 2027 2028)

for SEED in "${SEEDS[@]}"
do
    echo "Running seed ${SEED}"

    python src/train/finetune_dsdm.py \
        --config config/dsdm.yaml \
        --pretrained_ckpt outputs/checkpoints/dsdm_pretrain/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir outputs/checkpoints/dsdm_finetune/seed_${SEED} \
        --seed ${SEED}

    python src/train/generate_samples.py \
        --config config/dsdm.yaml \
        --ckpt outputs/checkpoints/dsdm_finetune/seed_${SEED}/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir data/generated/dsdm/seed_${SEED} \
        --samples_per_class 200 \
        --seed ${SEED}

    python src/train/train_cnn.py \
        --config config/classifier.yaml \
        --support_dir data/processed/support_set/seed_${SEED} \
        --generated_dir data/generated/dsdm/seed_${SEED} \
        --test_dir data/processed/test \
        --save_dir outputs/logs/cnn/dsdm/seed_${SEED} \
        --seed ${SEED}

    python src/eval/evaluate_generation.py \
        --real_dir data/processed/train \
        --generated_dir data/generated/dsdm/seed_${SEED} \
        --save_dir outputs/metrics/generation/dsdm/seed_${SEED} \
        --fs 12000
done