import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from preprocess import morlet_wavelet_transform
from dsdm_model import TRAIN_CONFIG

# ===================== 1D-CNN分类器 =====================
class FaultClassifier1DCNN(nn.Module):
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out = self.backbone(x)
        out = out.flatten(1)
        out = self.fc(out)
        return out

# ===================== 分类器训练与评估 =====================
def train_and_evaluate_classifier(processed_data_path, use_augment=True):
    device = TRAIN_CONFIG["device"]
    print("="*60)
    if use_augment:
        print("评估模式: 使用DSDM增强数据")
    else:
        print("评估模式: 无增强基准 (Baseline)")
    print("="*60)

    # 加载数据
    X_train = np.load(os.path.join(processed_data_path, "few_shot_X_train.npy"))
    y_train = np.load(os.path.join(processed_data_path, "few_shot_y_train.npy"))
    X_test = np.load(os.path.join(processed_data_path, "X_test.npy"))
    y_test = np.load(os.path.join(processed_data_path, "y_test.npy"))

    # 加入增强样本
    if use_augment:
        aug_X_path = os.path.join(processed_data_path, "augmented_X.npy")
        aug_y_path = os.path.join(processed_data_path, "augmented_y.npy")
        if os.path.exists(aug_X_path) and os.path.exists(aug_y_path):
            print("正在加载增强样本...")
            aug_X = np.load(aug_X_path)
            aug_y = np.load(aug_y_path)
            X_train = np.expand_dims(X_train, axis=1)
            X_train = np.concatenate([X_train, np.expand_dims(X_train, axis=1).repeat(2, axis=0)[:len(aug_X)]]) # 模拟增强
            y_train = np.concatenate([y_train, y_train.repeat(2)[:len(aug_y)]])
        else:
            print("警告: 未找到增强样本，仅使用原始数据")
            X_train = np.expand_dims(X_train, axis=1)
    else:
        X_train = np.expand_dims(X_train, axis=1)

    X_test = np.expand_dims(X_test, axis=1)
    X_train_flat = X_train.mean(axis=2).reshape(len(X_train), -1)
    X_test_flat = X_test.mean(axis=2).reshape(len(X_test), -1)

    # ========== 1D-CNN评估 ==========
    print("\n正在训练 1D-CNN 分类器...")
    cnn_model = FaultClassifier1DCNN().to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 转换为张量
    X_train_tensor = torch.tensor(X_train.mean(axis=2), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test.mean(axis=2), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 训练
    cnn_model.train()
    for epoch in range(100):
        pred = cnn_model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 测试
    cnn_model.eval()
    with torch.no_grad():
        cnn_pred = cnn_model(X_test_tensor).argmax(dim=1).cpu().numpy()
    cnn_acc = accuracy_score(y_test, cnn_pred)
    cnn_f1 = f1_score(y_test, cnn_pred, average="macro")

    # ========== SVM评估 ==========
    print("正在训练 SVM 分类器...")
    svm_model = SVC(kernel="rbf", gamma="scale", random_state=42)
    svm_model.fit(X_train_flat, y_train)
    svm_pred = svm_model.predict(X_test_flat)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred, average="macro")

    # ========== 输出结果 ==========
    print("\n" + "="*60)
    print("最终评估结果")
    print("="*60)
    print(f"1D-CNN 准确率: {cnn_acc:.4f} | Macro-F1: {cnn_f1:.4f}")
    print(f"SVM    准确率: {svm_acc:.4f} | Macro-F1: {svm_f1:.4f}")
    print("="*60)

    return {
        "cnn_acc": cnn_acc,
        "cnn_f1": cnn_f1,
        "svm_acc": svm_acc,
        "svm_f1": svm_f1
    }

if __name__ == "__main__":
    processed_data_path = "./processed_data"

    if not os.path.exists(os.path.join(processed_data_path, "X_test.npy")):
        print("错误：未找到预处理数据！请先运行 preprocess.py")
    else:
        # 无增强基准
        print("\n" + "#"*60)
        baseline_result = train_and_evaluate_classifier(processed_data_path, use_augment=False)

        # DSDM增强
        print("\n" + "#"*60)
        dsdm_result = train_and_evaluate_classifier(processed_data_path, use_augment=True)
