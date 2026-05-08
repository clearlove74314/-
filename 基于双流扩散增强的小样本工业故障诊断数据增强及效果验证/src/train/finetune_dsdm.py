import argparse
import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from src.models.dsdm import DSDM
from src.losses.diffusion_loss import DiffusionLoss
from src.losses.frequency_fidelity_loss import FrequencyFidelityLoss


def main():
    parser = argparse.ArgumentParser(description='Finetune DSDM model on few-shot data')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--pretrained_ckpt', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--support_dir', type=str, required=True, help='Support set directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--model_variant', type=str, default='dsdm', help='Model variant')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DSDM(config).to(device)
    model.load_state_dict(torch.load(args.pretrained_ckpt))
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.semantic_stream.parameters():
        param.requires_grad = True
    for param in model.cross_attention.parameters():
        param.requires_grad = True
    for param in model.diffusion_decoder.model[-1:].parameters():
        param.requires_grad = True
    
    x_support = np.load(os.path.join(args.support_dir, 'x_support.npy'))
    y_support = np.load(os.path.join(args.support_dir, 'y_support.npy'))
    
    x_support = torch.tensor(x_support, dtype=torch.float32).to(device)
    y_support = torch.tensor(y_support, dtype=torch.long).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'] * 10,
                           weight_decay=config['train']['weight_decay'])
    
    diffusion_loss_fn = DiffusionLoss()
    freq_loss_fn = FrequencyFidelityLoss()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(config['train']['finetune_epochs']):
        model.train()
        total_loss = 0.0
        
        for _ in range(10):
            permutation = torch.randperm(x_support.shape[0])
            x_shuffled = x_support[permutation]
            y_shuffled = y_support[permutation]
            
            optimizer.zero_grad()
            
            predicted_noise, target_noise, _ = model(x_shuffled, y_shuffled)
            
            diffusion_loss = diffusion_loss_fn(predicted_noise, target_noise)
            
            generated = model.generate(y_shuffled, num_samples=len(y_shuffled), device=device)
            freq_loss = freq_loss_fn(generated, x_shuffled)
            
            loss = diffusion_loss + config['loss']['lambda_freq'] * freq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        
        print(f"Finetune Epoch {epoch}: Loss {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'last_model.pth'))
    
    print("Finetuning completed successfully")


if __name__ == '__main__':
    main()