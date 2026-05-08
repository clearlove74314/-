import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_raw_data(raw_dir):
    categories = ['normal', 'inner_race', 'outer_race', 'gear_wear', 'rotor_imbalance']
    data = []
    labels = []
    
    for label, category in enumerate(categories):
        category_dir = os.path.join(raw_dir, category)
        if not os.path.exists(category_dir):
            continue
        
        for filename in sorted(os.listdir(category_dir)):
            filepath = os.path.join(category_dir, filename)
            if filename.endswith('.npy'):
                signal = np.load(filepath)
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath, header=None)
                signal = df.values.flatten()
            elif filename.endswith('.txt'):
                with open(filepath, 'r') as f:
                    signal = np.array([float(line.strip()) for line in f if line.strip()])
            elif filename.endswith('.mat'):
                mat_data = loadmat(filepath)
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if keys:
                    signal = mat_data[keys[0]].flatten()
                else:
                    continue
            else:
                continue
            
            data.append(signal)
            labels.append(label)
    
    return data, labels


def split_by_time_block(data, labels, train_ratio=0.8, val_ratio=0.1, gap=2048):
    all_windows = []
    all_labels = []
    all_indices = []
    
    for label, signal in zip(labels, data):
        num_samples = len(signal)
        total_ratio = train_ratio + val_ratio
        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * total_ratio)
        
        train_block = signal[:train_end]
        val_block = signal[train_end + gap : val_end]
        test_block = signal[val_end + gap :]
        
        for block, block_type in [(train_block, 'train'), (val_block, 'val'), (test_block, 'test')]:
            if len(block) < 2048:
                continue
            all_windows.append((block, label, block_type))
    
    train_data = []
    train_labels = []
    train_indices = []
    val_data = []
    val_labels = []
    val_indices = []
    test_data = []
    test_labels = []
    test_indices = []
    
    global_idx = 0
    for block, label, block_type in all_windows:
        num_windows = len(block) // 1024 - 1
        for i in range(num_windows):
            start = i * 1024
            end = start + 2048
            if end > len(block):
                continue
            
            window = block[start:end]
            indices = np.arange(start, end)
            
            if block_type == 'train':
                train_data.append(window)
                train_labels.append(label)
                train_indices.append(indices + global_idx)
            elif block_type == 'val':
                val_data.append(window)
                val_labels.append(label)
                val_indices.append(indices + global_idx)
            elif block_type == 'test':
                test_data.append(window)
                test_labels.append(label)
                test_indices.append(indices + global_idx)
        
        global_idx += len(block)
    
    return (np.array(train_data), np.array(train_labels), np.array(train_indices),
            np.array(val_data), np.array(val_labels), np.array(val_indices),
            np.array(test_data), np.array(test_labels), np.array(test_indices))


def main():
    parser = argparse.ArgumentParser(description='Data preprocessing for fault diagnosis')
    parser.add_argument('--raw_dir', type=str, required=True, help='Raw data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--fs', type=int, default=12000, help='Sampling frequency')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size')
    parser.add_argument('--stride', type=int, default=1024, help='Stride')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--gap', type=int, default=2048, help='Gap between blocks')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'index_record'), exist_ok=True)
    
    data, labels = load_raw_data(args.raw_dir)
    print(f"Loaded {len(data)} signal files")
    
    (x_train, y_train, train_indices,
     x_val, y_val, val_indices,
     x_test, y_test, test_indices) = split_by_time_block(
         data, labels, args.train_ratio, args.val_ratio, args.gap
     )
    
    print(f"Train samples: {len(x_train)}, Val samples: {len(x_val)}, Test samples: {len(x_test)}")
    
    np.save(os.path.join(args.save_dir, 'train', 'x_train.npy'), x_train)
    np.save(os.path.join(args.save_dir, 'train', 'y_train.npy'), y_train)
    np.save(os.path.join(args.save_dir, 'val', 'x_val.npy'), x_val)
    np.save(os.path.join(args.save_dir, 'val', 'y_val.npy'), y_val)
    np.save(os.path.join(args.save_dir, 'test', 'x_test.npy'), x_test)
    np.save(os.path.join(args.save_dir, 'test', 'y_test.npy'), y_test)
    np.save(os.path.join(args.save_dir, 'index_record', 'train_indices.npy'), train_indices)
    np.save(os.path.join(args.save_dir, 'index_record', 'val_indices.npy'), val_indices)
    np.save(os.path.join(args.save_dir, 'index_record', 'test_indices.npy'), test_indices)
    
    print("Preprocessing completed successfully")


if __name__ == '__main__':
    main()