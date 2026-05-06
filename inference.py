import torch
import numpy as np
import os
from tqdm import tqdm
from preprocess import FaultDataset, morlet_wavelet_transform
from torch.utils.data import DataLoader
from dsdm_model import DSDM, TRAIN_CONFIG

def generate_augmented_samples(model_path, processed_data_path, num_per_class=200, device=TRAIN_CONFIG["device"]):
    """生成增强样本"""
    print("="*50)
    print("开始生成增强样本...")
    print("="*50)

    # 加载模型
    print(f"1. 加载模型权重: {model_path}")
    model = DSDM(num_classes=TRAIN_CONFIG["num_classes"], timesteps=TRAIN_CONFIG["timesteps"]).to(device)
    best_model_path = os.path.join(model_path, "dsdm_best.pth")
    if not os.path.exists(best_model_path):
        print(f"警告: 未找到最优模型，尝试加载最终模型")
        best_model_path = os.path.join(model_path, "dsdm_final.pth")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 加载原始5-shot训练集
    print("2. 加载原始5-shot训练集...")
    train_dataset = FaultDataset(processed_data_path, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)

    augmented_samples = []
    augmented_labels = []

    print("3. 开始生成样本...")
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            cls = batch_y[0].item()
            print(f"   正在生成类别 {cls} 的增强样本...")

            # 每类生成num_per_class个样本
            gen_iterations = max(1, num_per_class // 5)
            for _ in tqdm(range(gen_iterations), desc=f"类别 {cls}"):
                gen_samples = model.generate_samples(batch_x, batch_y, device)
                augmented_samples.append(gen_samples.cpu().numpy())
                augmented_labels.append(np.full((5,), cls))

    # 合并生成的样本
    augmented_samples = np.concatenate(augmented_samples)
    augmented_labels = np.concatenate(augmented_labels)

    # 保存增强数据集
    np.save(os.path.join(processed_data_path, "augmented_X.npy"), augmented_samples)
    np.save(os.path.join(processed_data_path, "augmented_y.npy"), augmented_labels)

    print("="*50)
    print(f"增强样本生成完成！")
    print(f"  - 共生成 {len(augmented_samples)} 个样本")
    print(f"  - 保存路径: {processed_data_path}")
    print("="*50)
    return augmented_samples, augmented_labels

if __name__ == "__main__":
    model_path = "./dsdm_model"
    processed_data_path = "./processed_data"

    if not os.path.exists(os.path.join(processed_data_path, "few_shot_X_train.npy")):
        print("错误：未找到预处理数据！请先运行 preprocess.py")
    elif not os.path.exists(os.path.join(model_path, "dsdm_best.pth")) and not os.path.exists(os.path.join(model_path, "dsdm_final.pth")):
        print("错误：未找到模型权重！请先运行 train.py")
    else:
        aug_samples, aug_labels = generate_augmented_samples(model_path, processed_data_path, num_per_class=200)
