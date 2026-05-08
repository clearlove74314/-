import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


def evaluate_classifier(preds, labels):
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    
    try:
        auc = roc_auc_score(labels, np.eye(len(np.unique(labels)))[preds], multi_class='ovr')
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate classifier performance')
    parser.add_argument('--pred_path', type=str, required=True, help='Path to predictions')
    parser.add_argument('--label_path', type=str, required=True, help='Path to labels')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    args = parser.parse_args()
    
    preds = np.load(args.pred_path)
    labels = np.load(args.label_path)
    
    results = evaluate_classifier(preds, labels)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    df = pd.DataFrame([results])
    df.to_csv(os.path.join(args.save_dir, 'classification_metrics.csv'), index=False)
    
    np.save(os.path.join(args.save_dir, 'confusion_matrix.npy'), results['confusion_matrix'])
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"AUC: {results['auc']:.4f}")


if __name__ == '__main__':
    main()