import numpy as np
import os


def split_signal(signal, train_ratio=0.8, val_ratio=0.1, gap=2048):
    num_samples = len(signal)
    total_ratio = train_ratio + val_ratio
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * total_ratio)
    
    train_block = signal[:train_end]
    val_block = signal[train_end + gap : val_end]
    test_block = signal[val_end + gap :]
    
    return train_block, val_block, test_block


def sliding_window(signal, window_size=2048, stride=1024):
    windows = []
    indices = []
    num_windows = len(signal) // stride - 1
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        if end > len(signal):
            continue
        windows.append(signal[start:end])
        indices.append(np.arange(start, end))
    
    return np.array(windows), np.array(indices)


def split_by_time_block(data, labels, train_ratio=0.8, val_ratio=0.1, gap=2048,
                       window_size=2048, stride=1024):
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
    
    for label, signal in zip(labels, data):
        train_block, val_block, test_block = split_signal(signal, train_ratio, val_ratio, gap)
        
        if len(train_block) >= window_size:
            windows, idxs = sliding_window(train_block, window_size, stride)
            train_data.append(windows)
            train_labels.extend([label] * len(windows))
            train_indices.append(idxs + global_idx)
        
        if len(val_block) >= window_size:
            windows, idxs = sliding_window(val_block, window_size, stride)
            val_data.append(windows)
            val_labels.extend([label] * len(windows))
            val_indices.append(idxs + global_idx + len(train_block) + gap)
        
        if len(test_block) >= window_size:
            windows, idxs = sliding_window(test_block, window_size, stride)
            test_data.append(windows)
            test_labels.extend([label] * len(windows))
            test_indices.append(idxs + global_idx + len(train_block) + gap + len(val_block) + gap)
        
        global_idx += len(signal)
    
    if train_data:
        train_data = np.vstack(train_data)
        train_indices = np.vstack(train_indices)
    else:
        train_data = np.array([])
        train_indices = np.array([])
    
    if val_data:
        val_data = np.vstack(val_data)
        val_indices = np.vstack(val_indices)
    else:
        val_data = np.array([])
        val_indices = np.array([])
    
    if test_data:
        test_data = np.vstack(test_data)
        test_indices = np.vstack(test_indices)
    else:
        test_data = np.array([])
        test_indices = np.array([])
    
    return (train_data, np.array(train_labels), train_indices,
            val_data, np.array(val_labels), val_indices,
            test_data, np.array(test_labels), test_indices)


def save_split_data(save_dir, x_train, y_train, train_indices,
                   x_val, y_val, val_indices,
                   x_test, y_test, test_indices):
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'index_record'), exist_ok=True)
    
    np.save(os.path.join(save_dir, 'train', 'x_train.npy'), x_train)
    np.save(os.path.join(save_dir, 'train', 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'val', 'x_val.npy'), x_val)
    np.save(os.path.join(save_dir, 'val', 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'test', 'x_test.npy'), x_test)
    np.save(os.path.join(save_dir, 'test', 'y_test.npy'), y_test)
    np.save(os.path.join(save_dir, 'index_record', 'train_indices.npy'), train_indices)
    np.save(os.path.join(save_dir, 'index_record', 'val_indices.npy'), val_indices)
    np.save(os.path.join(save_dir, 'index_record', 'test_indices.npy'), test_indices)