import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(data_dir, save_path, num_classes=5):
    class_names = ['Normal', 'Inner Race', 'Outer Race', 'Gear Wear', 'Rotor Imbalance']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 10))
    
    for cls in range(num_classes):
        indices = np.where(y_train == cls)[0]
        sample = x_train[indices[0]]
        
        axes[cls].plot(sample[:500], color=colors[cls])
        axes[cls].set_title(class_names[cls])
        axes[cls].set_xlabel('Sample')
        axes[cls].set_ylabel('Amplitude')
        axes[cls].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot waveform visualization')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--save_path', type=str, required=True, help='Save path for plot')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    plot_waveform(args.data_dir, args.save_path, args.num_classes)
    
    print(f"Waveform plot saved to {args.save_path}")


if __name__ == '__main__':
    main()