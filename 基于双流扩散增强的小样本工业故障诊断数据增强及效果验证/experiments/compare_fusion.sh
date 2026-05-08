#!/bin/bash

SEEDS=(2024 2025 2026 2027 2028)

for SEED in "${SEEDS[@]}"
do
    echo "=== Processing seed ${SEED} ==="
    
    echo "--- DSDM-Add ---"
    python src/train/finetune_dsdm.py \
        --config config/dsdm.yaml \
        --model_variant add \
        --pretrained_ckpt outputs/checkpoints/dsdm_pretrain/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir outputs/checkpoints/dsdm_add/seed_${SEED} \
        --seed ${SEED}
    
    python src/train/generate_samples.py \
        --config config/dsdm.yaml \
        --model_variant add \
        --ckpt outputs/checkpoints/dsdm_add/seed_${SEED}/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir data/generated/dsdm_add/seed_${SEED} \
        --samples_per_class 200 \
        --seed ${SEED}
    
    python src/train/train_cnn.py \
        --config config/classifier.yaml \
        --support_dir data/processed/support_set/seed_${SEED} \
        --generated_dir data/generated/dsdm_add/seed_${SEED} \
        --test_dir data/processed/test \
        --save_dir outputs/logs/cnn/dsdm_add/seed_${SEED} \
        --seed ${SEED}
    
    echo "--- DSDM-Concat ---"
    python src/train/finetune_dsdm.py \
        --config config/dsdm.yaml \
        --model_variant concat \
        --pretrained_ckpt outputs/checkpoints/dsdm_pretrain/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir outputs/checkpoints/dsdm_concat/seed_${SEED} \
        --seed ${SEED}
    
    python src/train/generate_samples.py \
        --config config/dsdm.yaml \
        --model_variant concat \
        --ckpt outputs/checkpoints/dsdm_concat/seed_${SEED}/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir data/generated/dsdm_concat/seed_${SEED} \
        --samples_per_class 200 \
        --seed ${SEED}
    
    python src/train/train_cnn.py \
        --config config/classifier.yaml \
        --support_dir data/processed/support_set/seed_${SEED} \
        --generated_dir data/generated/dsdm_concat/seed_${SEED} \
        --test_dir data/processed/test \
        --save_dir outputs/logs/cnn/dsdm_concat/seed_${SEED} \
        --seed ${SEED}
    
    echo "--- DSDM-Linear ---"
    python src/train/finetune_dsdm.py \
        --config config/dsdm.yaml \
        --model_variant linear \
        --noise_schedule linear \
        --pretrained_ckpt outputs/checkpoints/dsdm_pretrain/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir outputs/checkpoints/dsdm_linear/seed_${SEED} \
        --seed ${SEED}
    
    python src/train/generate_samples.py \
        --config config/dsdm.yaml \
        --model_variant linear \
        --noise_schedule linear \
        --ckpt outputs/checkpoints/dsdm_linear/seed_${SEED}/best_model.pth \
        --support_dir data/processed/support_set/seed_${SEED} \
        --save_dir data/generated/dsdm_linear/seed_${SEED} \
        --samples_per_class 200 \
        --seed ${SEED}
    
    python src/train/train_cnn.py \
        --config config/classifier.yaml \
        --support_dir data/processed/support_set/seed_${SEED} \
        --generated_dir data/generated/dsdm_linear/seed_${SEED} \
        --test_dir data/processed/test \
        --save_dir outputs/logs/cnn/dsdm_linear/seed_${SEED} \
        --seed ${SEED}
done