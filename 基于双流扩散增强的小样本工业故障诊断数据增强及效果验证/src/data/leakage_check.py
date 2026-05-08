import argparse
import os
import numpy as np


def check_leakage(index_dir):
    train_indices = np.load(os.path.join(index_dir, 'train_indices.npy'))
    val_indices = np.load(os.path.join(index_dir, 'val_indices.npy'))
    test_indices = np.load(os.path.join(index_dir, 'test_indices.npy'))
    
    train_set = set()
    for window in train_indices:
        train_set.update(window.tolist())
    
    val_set = set()
    for window in val_indices:
        val_set.update(window.tolist())
    
    test_set = set()
    for window in test_indices:
        test_set.update(window.tolist())
    
    train_val_intersect = len(train_set.intersection(val_set))
    train_test_intersect = len(train_set.intersection(test_set))
    val_test_intersect = len(val_set.intersection(test_set))
    
    print("[Leakage Check]")
    print(f"train ∩ val  : {train_val_intersect}")
    print(f"train ∩ test : {train_test_intersect}")
    print(f"val ∩ test   : {val_test_intersect}")
    print()
    
    if train_val_intersect == 0 and train_test_intersect == 0 and val_test_intersect == 0:
        print("Result: No sampling-point leakage detected.")
        return True
    else:
        print("Result: Sampling-point leakage detected!")
        return False


def main():
    parser = argparse.ArgumentParser(description='Check for data leakage between train/val/test sets')
    parser.add_argument('--index_dir', type=str, required=True, help='Directory containing index files')
    args = parser.parse_args()
    
    check_leakage(args.index_dir)


if __name__ == '__main__':
    main()