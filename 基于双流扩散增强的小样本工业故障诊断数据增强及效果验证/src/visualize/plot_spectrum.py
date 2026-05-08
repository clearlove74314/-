import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram


def plot_spectrum(data_dir, fs, save_path, compare_real_generated=False, generated_dir=None):
    class_names = ['Normal', 'Inner Race', 'Outer Race', 'Gear Wear', 'Rotor Imbalance']
    
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    if compare_real_generated:
        x_gen = np.load(os.path.join(generated_dir, 'x_generated.npy'))
        y_gen = np.load(os.path.join(generated_dir, 'y_generated.npy'))
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    
    for cls in range(5):
        indices = np.where(y_train == cls)[0]
        sample = x_train[indices[0]]
        
        freqs, psd = periodogram(sample, fs=fs)
        psd = 10 * np.log10(psd)
        
        axes[cls].plot(freqs, psd, label='Real', color='blue')
        
        if compare_real_generated:
            gen_indices = np.where(y_gen == cls)[0]
            gen_sample = x_gen[gen_indices[0]]
            gen_freqs, gen_psd = periodogram(gen_sample, fs=fs)
            gen_psd = 10 * np.log10(gen_psd)
            axes[cls].plot(gen_freqs, gen_psd, label='Generated', color='red', alpha=0.7)
        
        axes[cls].set_title(class_names[cls])
        axes[cls].set_xlabel('Frequency (Hz)')
        axes[cls].set_ylabel('Power Spectral Density (dB)')
        axes[cls].set_xlim(0, 2000)
        axes[cls].grid(True)
        if compare_real_generated:
            axes[cls].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot spectrum visualization')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--fs', type=int, default=12000, help='Sampling frequency')
    parser.add_argument('--save_path', type=str, required=True, help='Save path for plot')
    parser.add_argument('--compare_real_generated', action='store_true', help='Compare real and generated')
    parser.add_argument('--generated_dir', type=str, help='Generated data directory')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    plot_spectrum(args.data_dir, args.fs, args.save_path, args.compare_real_generated, args.generated_dir)
    
    print(f"Spectrum plot saved to {args.save_path}")


if __name__ == '__main__':
    main()