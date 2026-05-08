import argparse
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from src.models.dsdm import DSDM


def main():
    parser = argparse.ArgumentParser(description='Generate samples using DSDM')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--support_dir', type=str, required=True, help='Support set directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--samples_per_class', type=int, default=200, help='Number of samples per class')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--model_variant', type=str, default='dsdm', help='Model variant')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DSDM(config).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    
    num_classes = config['model']['num_classes']
    
    generated_samples = []
    generated_labels = []
    
    with torch.no_grad():
        for cls in tqdm(range(num_classes), desc="Generating samples"):
            labels = torch.tensor([cls] * args.samples_per_class, dtype=torch.long).to(device)
            samples = model.generate(labels, num_samples=args.samples_per_class, device=device)
            
            generated_samples.append(samples.cpu().numpy())
            generated_labels.extend([cls] * args.samples_per_class)
    
    generated_samples = np.vstack(generated_samples)
    generated_labels = np.array(generated_labels)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    np.save(os.path.join(args.save_dir, 'x_generated.npy'), generated_samples)
    np.save(os.path.join(args.save_dir, 'y_generated.npy'), generated_labels)
    
    print(f"Generated {len(generated_samples)} samples")


if __name__ == '__main__':
    main()