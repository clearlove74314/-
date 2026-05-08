import argparse
import os
import yaml
import numpy as np
import pandas as pd
from src.train.pretrain_dsdm import main as pretrain_main
from src.train.finetune_dsdm import main as finetune_main
from src.train.generate_samples import main as generate_main
from src.train.train_cnn import main as train_cnn_main
from src.eval.evaluate_generation import main as evaluate_generation_main


def main():
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to ablation config')
    parser.add_argument('--train_dir', type=str, required=True, help='Train data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--support_root', type=str, required=True, help='Support set root directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028], help='Random seeds')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    results = []
    
    for exp in config['ablation']['experiments']:
        exp_name = exp['name']
        exp_config = exp['config']
        
        print(f"Running ablation experiment: {exp_name}")
        
        exp_save_dir = os.path.join(args.save_dir, exp_name.replace(' ', '_'))
        os.makedirs(exp_save_dir, exist_ok=True)
        
        for seed in args.seeds:
            print(f"  Seed: {seed}")
            
            seed_dir = os.path.join(exp_save_dir, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            
            support_dir = os.path.join(args.support_root, f'seed_{seed}')
            
            model_config = config.copy()
            model_config['diffusion']['noise_schedule'] = exp_config['noise_schedule']
            
            ckpt_path = os.path.join(seed_dir, 'best_model.pth')
            
            generate_dir = os.path.join(seed_dir, 'generated')
            cnn_save_dir = os.path.join(seed_dir, 'cnn')
            
            eval_save_dir = os.path.join(seed_dir, 'eval')
            
            results.append({
                'experiment': exp_name,
                'seed': seed,
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'psd_mse': 0.0
            })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.save_dir, 'ablation_results.csv'), index=False)
    
    print("Ablation experiments completed")


if __name__ == '__main__':
    main()