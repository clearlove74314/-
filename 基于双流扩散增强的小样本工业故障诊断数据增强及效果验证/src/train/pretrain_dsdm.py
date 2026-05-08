import argparse
import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from src.models.dsdm import DSDM
from src.losses.diffusion_loss import DiffusionLoss


def main():
    parser = argparse.ArgumentParser(description='Pretrain DSDM model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--train_dir', type=str, required=True, help='Train data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_train = np.load(os.path.join(args.train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(args.train_dir, 'y_train.npy'))
    x_val = np.load(os.path.join(args.val_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(args.val_dir, 'y_val.npy'))
    
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    model = DSDM(config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], 
                           weight_decay=config['train']['weight_decay'])
    
    loss_fn = DiffusionLoss()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    train_log = []
    
    for epoch in range(config['train']['pretrain_epochs']):
        model.train()
        train_loss = 0.0
        
        permutation = torch.randperm(x_train.shape[0])
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        for i in range(0, x_train.shape[0], config['train']['batch_size']):
            x_batch = x_train_shuffled[i:i+config['train']['batch_size']]
            y_batch = y_train_shuffled[i:i+config['train']['batch_size']]
            
            optimizer.zero_grad()
            
            predicted_noise, target_noise, _ = model(x_batch, y_batch)
            loss = loss_fn(predicted_noise, target_noise)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.shape[0]
        
        train_loss /= x_train.shape[0]
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, x_val.shape[0], config['train']['batch_size']):
                x_batch = x_val[i:i+config['train']['batch_size']]
                y_batch = y_val[i:i+config['train']['batch_size']]
                
                predicted_noise, target_noise, _ = model(x_batch, y_batch)
                loss = loss_fn(predicted_noise, target_noise)
                
                val_loss += loss.item() * x_batch.shape[0]
        
        val_loss /= x_val.shape[0]
        
        train_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'last_model.pth'))
    
    np.save(os.path.join(args.save_dir, 'pretrain_log.npy'), np.array(train_log))
    
    print("Pretraining completed successfully")


if __name__ == '__main__':
    main()