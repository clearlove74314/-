import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(result_csv, metric, save_path):
    df = pd.read_csv(result_csv)
    
    if 'experiment' in df.columns:
        means = df.groupby('experiment')[metric].mean().values
        stds = df.groupby('experiment')[metric].std().values
        labels = df.groupby('experiment').groups.keys()
    else:
        means = df.groupby('method')[metric].mean().values
        stds = df.groupby('method')[metric].std().values
        labels = df.groupby('method').groups.keys()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, means, yerr=stds, capsize=5)
    
    ax.set_xlabel('Method')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot classification results')
    parser.add_argument('--result_csv', type=str, required=True, help='Path to results CSV')
    parser.add_argument('--metric', type=str, default='accuracy', help='Metric to plot')
    parser.add_argument('--save_path', type=str, required=True, help='Save path for plot')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    plot_results(args.result_csv, args.metric, args.save_path)
    
    print(f"Results plot saved to {args.save_path}")


if __name__ == '__main__':
    main()