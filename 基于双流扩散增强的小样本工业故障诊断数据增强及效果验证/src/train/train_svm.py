import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.models.svm_classifier import SVMClassifier


def main():
    parser = argparse.ArgumentParser(description='Train SVM classifier')
    parser.add_argument('--support_dir', type=str, required=True, help='Support set directory')
    parser.add_argument('--generated_dir', type=str, help='Generated samples directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    x_support = np.load(os.path.join(args.support_dir, 'x_support.npy'))
    y_support = np.load(os.path.join(args.support_dir, 'y_support.npy'))
    
    if args.generated_dir:
        x_generated = np.load(os.path.join(args.generated_dir, 'x_generated.npy'))
        y_generated = np.load(os.path.join(args.generated_dir, 'y_generated.npy'))
        x_train = np.vstack([x_support, x_generated])
        y_train = np.hstack([y_support, y_generated])
    else:
        x_train = x_support
        y_train = y_support
    
    x_test = np.load(os.path.join(args.test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(args.test_dir, 'y_test.npy'))
    
    clf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(x_train, y_train)
    
    preds = clf.predict(x_test)
    
    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')
    cm = confusion_matrix(y_test, preds)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    clf.save(os.path.join(args.save_dir, 'svm_model.pkl'))
    
    with open(os.path.join(args.save_dir, 'svm_metrics.csv'), 'w') as f:
        f.write('metric,value\n')
        f.write(f'accuracy,{accuracy}\n')
        f.write(f'macro_f1,{macro_f1}\n')
    
    np.save(os.path.join(args.save_dir, 'svm_confusion_matrix.npy'), cm)
    
    print(f"SVM Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")


if __name__ == '__main__':
    main()