import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.signal import periodogram


def calculate_psd_mse(real, generated, fs=12000):
    psd_real, _ = periodogram(real, fs=fs, axis=-1)
    psd_gen, _ = periodogram(generated, fs=fs, axis=-1)
    return np.mean((psd_real - psd_gen) ** 2)


def calculate_mmd(real, generated):
    real_flat = real.reshape(real.shape[0], -1)
    gen_flat = generated.reshape(generated.shape[0], -1)
    
    xx = pairwise_distances(real_flat, metric='rbf')
    yy = pairwise_distances(gen_flat, metric='rbf')
    xy = pairwise_distances(real_flat, gen_flat, metric='rbf')
    
    return xx.mean() + yy.mean() - 2 * xy.mean()


def calculate_dtw(real, generated):
    distances = []
    for r, g in zip(real, generated):
        dtw_dist = dtw_distance(r, g)
        distances.append(dtw_dist)
    return np.mean(distances)


def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[0, :] = np.inf
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n, m]


def calculate_fd(real, generated):
    real_flat = real.reshape(real.shape[0], -1)
    gen_flat = generated.reshape(generated.shape[0], -1)
    
    mu_real = np.mean(real_flat, axis=0)
    mu_gen = np.mean(gen_flat, axis=0)
    cov_real = np.cov(real_flat.T)
    cov_gen = np.cov(gen_flat.T)
    
    diff = mu_real - mu_gen
    cov_mean = (cov_real + cov_gen) / 2
    
    fd = diff @ np.linalg.pinv(cov_mean) @ diff + np.trace(cov_real + cov_gen - 2 * np.sqrt(cov_real @ cov_gen))
    return np.real(fd)


def calculate_ks_test(real, generated):
    passed = 0
    for i in range(min(len(real), len(generated))):
        r = real[i]
        g = generated[i]
        _, p = stats.ks_2samp(r, g)
        if p > 0.05:
            passed += 1
    return passed / min(len(real), len(generated))


def main():
    parser = argparse.ArgumentParser(description='Evaluate generation quality')
    parser.add_argument('--real_dir', type=str, required=True, help='Real data directory')
    parser.add_argument('--generated_dir', type=str, required=True, help='Generated data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--fs', type=int, default=12000, help='Sampling frequency')
    args = parser.parse_args()
    
    x_real = np.load(os.path.join(args.real_dir, 'x_train.npy'))
    x_gen = np.load(os.path.join(generated_dir, 'x_generated.npy'))
    
    min_samples = min(len(x_real), len(x_gen))
    x_real = x_real[:min_samples]
    x_gen = x_gen[:min_samples]
    
    psd_mse = calculate_psd_mse(x_real, x_gen, args.fs)
    mmd = calculate_mmd(x_real, x_gen)
    dtw = calculate_dtw(x_real, x_gen)
    fd = calculate_fd(x_real, x_gen)
    ks_pass_rate = calculate_ks_test(x_real, x_gen)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = {
        'psd_mse': psd_mse,
        'mmd': mmd,
        'dtw': dtw,
        'fd': fd,
        'ks_pass_rate': ks_pass_rate
    }
    
    df = pd.DataFrame([results])
    df.to_csv(os.path.join(args.save_dir, 'generation_quality_summary.csv'), index=False)
    
    print(f"PSD-MSE: {psd_mse:.4f}")
    print(f"MMD: {mmd:.4f}")
    print(f"DTW: {dtw:.4f}")
    print(f"FD: {fd:.4f}")
    print(f"KS Pass Rate: {ks_pass_rate:.2%}")


if __name__ == '__main__':
    main()