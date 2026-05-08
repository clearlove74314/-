import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from src.models.dsdm import DSDM


def plot_attention(ckpt_path, support_dir, save_path):
    config_path = 'config/dsdm.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = DSDM(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    
    x_support = np.load(os.path.join(support_dir, 'x_support.npy'))
    y_support = np.load(os.path.join(support_dir, 'y_support.npy'))
    
    x_tensor = torch.tensor(x_support[:5], dtype=torch.float32)
    y_tensor = torch.tensor(y_support[:5], dtype=torch.long)
    
    with torch.no_grad():
        _, _, attn_weights = model(x_tensor, y_tensor)
    
    attn_weights = attn_weights.mean(dim=1).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_weights, cmap='viridis')
    
    ax.set_xlabel('Semantic Features')
    ax.set_ylabel('Feature Features')
    ax.set_title('Cross-Attention Weights')
    
    plt.colorbar(im)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot attention weights visualization')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--support_dir', type=str, required=True, help='Support set directory')
    parser.add_argument('--save_path', type=str, required=True, help='Save path for plot')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    plot_attention(args.ckpt, args.support_dir, args.save_path)
    
    print(f"Attention weights plot saved to {args.save_path}")


if __name__ == '__main__':
    main()