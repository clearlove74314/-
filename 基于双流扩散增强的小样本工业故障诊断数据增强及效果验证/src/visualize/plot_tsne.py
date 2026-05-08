import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(real_dir, generated_dir, save_path):
    x_real = np.load(os.path.join(real_dir, 'x_train.npy'))
    y_real = np.load(os.path.join(real_dir, 'y_train.npy'))
    
    x_gen = np.load(os.path.join(generated_dir, 'x_generated.npy'))
    y_gen = np.load(os.path.join(generated_dir, 'y_generated.npy'))
    
    x_combined = np.vstack([x_real[:100], x_gen[:100]])
    y_combined = np.hstack([y_real[:100], y_gen[:100]])
    labels = np.array(['Real'] * 100 + ['Generated'] * 100)
    
    x_flat = x_combined.reshape(x_combined.shape[0], -1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(x_flat)
    
    class_names = ['Normal', 'Inner Race', 'Outer Race', 'Gear Wear', 'Rotor Imbalance']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls in range(5):
        mask = (y_combined == cls) & (labels == 'Real')
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   label=f'{class_names[cls]} (Real)', 
                   color=colors[cls], alpha=0.6)
        
        mask = (y_combined == cls) & (labels == 'Generated')
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   label=f'{class_names[cls]} (Generated)', 
                   color=colors[cls], alpha=0.6, marker='x')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('t-SNE Visualization')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot t-SNE visualization')
    parser.add_argument('--real_dir', type=str, required=True, help='Real data directory')
    parser.add_argument('--generated_dir', type=str, required=True, help='Generated data directory')
    parser.add_argument('--save_path', type=str, required=True, help='Save path for plot')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    plot_tsne(args.real_dir, args.generated_dir, args.save_path)
    
    print(f"t-SNE plot saved to {args.save_path}")


if __name__ == '__main__':
    main()