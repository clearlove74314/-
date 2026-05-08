import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix plot')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to predictions')
    parser.add_argument('--label_path', type=str, required=True, help='Path to labels')
    parser.add_argument('--save_path', type=str, required=True, help='Save path for plot')
    args = parser.parse_args()
    
    preds = np.load(args.pred_path)
    labels = np.load(args.label_path)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    
    class_names = ['Normal', 'Inner Race', 'Outer Race', 'Gear Wear', 'Rotor Imbalance']
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    plot_confusion_matrix(cm, class_names, args.save_path)
    
    print(f"Confusion matrix saved to {args.save_path}")


if __name__ == '__main__':
    main()