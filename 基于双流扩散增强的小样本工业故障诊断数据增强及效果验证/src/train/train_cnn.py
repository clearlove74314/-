import argparse
import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from src.models.cnn1d import CNN1D
from src.losses.classification_loss import ClassificationLoss


def main():
    parser = argparse.ArgumentParser(description='Train 1-D CNN classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--support_dir', type=str, required=True, help='Support set directory')
    parser.add_argument('--generated_dir', type=str, help='Generated samples directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_support = np.load(os.path.join(args.support_dir, 'x_support.npy'))
    y_support = np.load(os.path.join(args.support_dir, 'y_support.npy'))
    
    if args.generated_dir:
        x_generated = np.load(os.path.join(args.generated_dir, 'x_generated.npy'))
        y_generated = np.load(os.path.join(args.generated_dir, 'y_generated.npy'))
        x_train = np.vstack([x_support, x_generated])
        y_train = np.hstack([y_support, y_generated])
    else:
        x_train = x_support
        y_train = y_support
    
    x_test = np.load(os.path.join(args.test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(args.test_dir, 'y_test.npy'))
    
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    model = CNN1D(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'],
                           weight_decay=config['train']['weight_decay'])
    
    loss_fn = ClassificationLoss(num_classes=config['classifier']['num_classes'])
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    best_acc = 0.0
    train_log = []
    
    for epoch in range(config['train']['epochs']):
        model.train()
        train_loss = 0.0
        
        permutation = torch.randperm(x_train.shape[0])
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        for i in range(0, x_train.shape[0], config['train']['batch_size']):
            x_batch = x_train_shuffled[i:i+config['train']['batch_size']]
            y_batch = y_train_shuffled[i:i+config['train']['batch_size']]
            
            optimizer.zero_grad()
            
            logits, _ = model(x_batch)
            loss = loss_fn(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.shape[0]
        
        train_loss /= x_train.shape[0]
        
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_test)
            preds = torch.argmax(logits, dim=1)
            
            accuracy = accuracy_score(y_test.cpu().numpy(), preds.cpu().numpy())
            macro_f1 = f1_score(y_test.cpu().numpy(), preds.cpu().numpy(), average='macro')
            
        train_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'accuracy': accuracy,
            'macro_f1': macro_f1
        })
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.6f}, Accuracy {accuracy:.4f}, Macro-F1 {macro_f1:.4f}")
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_cnn.pth'))
            np.save(os.path.join(args.save_dir, 'predictions.npy'), preds.cpu().numpy())
            np.save(os.path.join(args.save_dir, 'labels.npy'), y_test.cpu().numpy())
    
    np.save(os.path.join(args.save_dir, 'train_log.npy'), np.array(train_log))
    
    print(f"Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()