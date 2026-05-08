#!/bin/bash

SEEDS=(2024 2025 2026 2027 2028)

for SEED in "${SEEDS[@]}"
do
    echo "=== Processing seed ${SEED} ==="
    
    echo "--- Cosine Schedule (DSDM) ---"
    python src/train/finetune_dsdm.py \
        --config config/dsdm.yaml \
        --noise_schedule cosine \
        --pretrained_ckpt outputs/checkpoints/dsdm_pretrain/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir outputs/checkpoints/dsdm_cosine/seed_${SEED} \
        --seed ${SEED}
    
    python src/train/generate_samples.py \
        --config config/dsdm.yaml \
        --noise_schedule cosine \
        --ckpt outputs/checkpoints/dsdm_cosine/seed_${SEED}/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir data/generated/dsdm_cosine/seed_${SEED} \
        --samples_per_class 200 \
        --seed ${SEED}
    
    python src/train/train_cnn.py \
        --config config/classifier.yaml \
        --support_dir data/processed/support_set/seed_${SEED} \
        --generated_dir data/generated/dsdm_cosine/seed_${SEED} \
        --test_dir data/processed/test \
        --save_dir outputs/logs/cnn/dsdm_cosine/seed_${SEED} \
        --seed ${SEED}
    
    echo "--- Linear Schedule ---"
    python src/train/finetune_dsdm.py \
        --config config/dsdm.yaml \
        --noise_schedule linear \
        --pretrained_ckpt outputs/checkpoints/dsdm_pretrain/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir outputs/checkpoints/dsdm_linear_sched/seed_${SEED} \
        --seed ${SEED}
    
    python src/train/generate_samples.py \
        --config config/dsdm.yaml \
        --noise_schedule linear \
        --ckpt outputs/checkpoints/dsdm_linear_sched/seed_${SEED}/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir data/generated/dsdm_linear_sched/seed_${SEED} \
        --samples_per_class 200 \
        --seed ${SEED}
    
    python src/train/train_cnn.py \
        --config config/classifier.yaml \
        --support_dir data/processed/support_set/seed_${SEED} \
        --generated_dir data/generated/dsdm_linear_sched/seed_${SEED} \
        --test_dir data/processed/test \
        --save_dir outputs/logs/cnn/dsdm_linear_sched/seed_${SEED} \
        --seed ${SEED}
done