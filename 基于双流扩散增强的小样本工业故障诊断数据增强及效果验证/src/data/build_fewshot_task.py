import argparse
import os
import numpy as np


def build_fewshot_task(x_train, y_train, indices, num_classes=5, shot=5, seed=2024):
    np.random.seed(seed)
    
    support_indices = []
    support_labels = []
    support_sample_indices = []
    
    for cls in range(num_classes):
        cls_indices = np.where(y_train == cls)[0]
        selected = np.random.choice(cls_indices, size=shot, replace=False)
        
        support_indices.extend(selected)
        support_labels.extend([cls] * shot)
        support_sample_indices.extend(indices[selected])
    
    support_indices = np.array(support_indices)
    support_labels = np.array(support_labels)
    
    x_support = x_train[support_indices]
    support_sample_indices = np.array(support_sample_indices)
    
    return x_support, support_labels, support_sample_indices


def main():
    parser = argparse.ArgumentParser(description='Build few-shot support set')
    parser.add_argument('--train_dir', type=str, required=True, help='Train data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--shot', type=int, default=5, help='Number of shots')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026, 2027, 2028], help='Random seeds')
    args = parser.parse_args()
    
    x_train = np.load(os.path.join(args.train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(args.train_dir, 'y_train.npy'))
    indices = np.load(os.path.join(args.train_dir.replace('train', 'index_record'), 'train_indices.npy'))
    
    for seed in args.seeds:
        seed_dir = os.path.join(args.save_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        
        x_support, y_support, support_indices = build_fewshot_task(
            x_train, y_train, indices, args.num_classes, args.shot, seed
        )
        
        np.save(os.path.join(seed_dir, 'x_support.npy'), x_support)
        np.save(os.path.join(seed_dir, 'y_support.npy'), y_support)
        np.save(os.path.join(seed_dir, 'support_indices.npy'), support_indices)
        
        print(f"Created support set for seed {seed}: {len(x_support)} samples")
    
    print("Few-shot support sets created successfully")


if __name__ == '__main__':
    main()