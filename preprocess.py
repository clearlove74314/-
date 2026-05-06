import numpy as np
import pywt
import os
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# ===================== 核心参数配置 =====================
SAMPLE_RATE = 12000
SIGNAL_LENGTH = 1024
OVERLAP_RATIO = 0.5
WAVELET_SCALE = 256
CLASS_NUM = 5
SHOT_NUM = 5
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== 1. 原始数据加载函数 =====================
def load_raw_data(data_root):
    data_list = []
    label_list = []

    class_folders = {
        0: "normal",
        1: "bearing_inner",
        2: "bearing_outer",
        3: "gear_wear",
        4: "rotor_unbalance"
    }

    for label, folder_name in class_folders.items():
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在 {folder_path}，跳过该类别")
            continue
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".mat"):
                try:
                    mat_data = loadmat(os.path.join(folder_path, file_name))
                    signal_keys = ["gs", "bearing", "vibration", "X097_DE_time", "X100_DE_time"]
                    raw_signal = None
                    for key in signal_keys:
                        if key in mat_data:
                            raw_signal = mat_data[key].flatten()
                            break
                    if raw_signal is not None:
                        data_list.append(raw_signal)
                        label_list.append(label)
                except Exception as e:
                    print(f"读取文件 {file_name} 失败: {e}")

    if not data_list:
        raise ValueError("未加载到任何数据，请检查raw_dataset目录结构！")

    return data_list, label_list

# ===================== 2. 重叠采样与信号分段 =====================
def overlap_sampling(signal_list, label_list, signal_len=SIGNAL_LENGTH, overlap=OVERLAP_RATIO):
    step = int(signal_len * (1 - overlap))
    signals = []
    labels = []

    for signal, label in zip(signal_list, label_list):
        start_idx = 0
        while start_idx + signal_len <= len(signal):
            seg_signal = signal[start_idx:start_idx+signal_len]
            signals.append(seg_signal)
            labels.append(label)
            start_idx += step

    return np.array(signals), np.array(labels)

# ===================== 3. Z-score归一化 =====================
def z_score_normalize(signals):
    scaler = StandardScaler()
    normalized_signals = scaler.fit_transform(signals.T).T
    return normalized_signals, scaler

# ===================== 4. Morlet小波时频变换 =====================
def morlet_wavelet_transform(signal, scales=WAVELET_SCALE, sample_rate=SAMPLE_RATE):
    frequencies = np.linspace(1, sample_rate//2, scales)
    [cwt_matrix, _] = pywt.cwt(signal, frequencies, "morl", 1/sample_rate)
    cwt_matrix = (cwt_matrix - cwt_matrix.min()) / (cwt_matrix.max() - cwt_matrix.min() + 1e-8)
    return cwt_matrix.astype(np.float32)

# ===================== 5. 数据集批量预处理 =====================
def preprocess_pipeline(data_root, save_path="./processed_data"):
    print("="*50)
    print("开始数据预处理...")
    print("="*50)

    # 1. 加载原始数据
    print("1. 正在加载原始数据...")
    raw_signals, raw_labels = load_raw_data(data_root)
    print(f"   加载到 {len(raw_signals)} 条原始信号")

    # 2. 重叠采样
    print("2. 正在进行重叠采样...")
    seg_signals, seg_labels = overlap_sampling(raw_signals, raw_labels)
    print(f"   重叠采样后得到 {len(seg_signals)} 个样本")

    # 3. Z-score归一化
    print("3. 正在进行Z-score归一化...")
    norm_signals, scaler = z_score_normalize(seg_signals)

    # 4. 小波时频变换
    print("4. 正在生成时频图谱 (这可能需要一些时间)...")
    time_freq_data = []
    for i, signal in enumerate(norm_signals):
        if i % 1000 == 0:
            print(f"   已处理 {i}/{len(norm_signals)}...")
        tf_map = morlet_wavelet_transform(signal)
        time_freq_data.append(tf_map)
    time_freq_data = np.array(time_freq_data)

    # 5. 数据集划分 8:1:1
    print("5. 正在划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        time_freq_data, seg_labels, test_size=0.2, random_state=SEED, stratify=seg_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    # 6. 构建5-shot小样本训练集
    print("6. 正在构建5-shot小样本训练集...")
    few_shot_X_train = []
    few_shot_y_train = []
    for cls in range(CLASS_NUM):
        cls_idx = np.where(y_train == cls)[0]
        if len(cls_idx) < SHOT_NUM:
            raise ValueError(f"类别 {cls} 的样本数不足 {SHOT_NUM} 个！")
        selected_idx = np.random.choice(cls_idx, SHOT_NUM, replace=False)
        few_shot_X_train.append(X_train[selected_idx])
        few_shot_y_train.append(y_train[selected_idx])
    few_shot_X_train = np.concatenate(few_shot_X_train)
    few_shot_y_train = np.concatenate(few_shot_y_train)

    # 保存预处理数据
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "few_shot_X_train.npy"), few_shot_X_train)
    np.save(os.path.join(save_path, "few_shot_y_train.npy"), few_shot_y_train)
    np.save(os.path.join(save_path, "X_val.npy"), X_val)
    np.save(os.path.join(save_path, "y_val.npy"), y_val)
    np.save(os.path.join(save_path, "X_test.npy"), X_test)
    np.save(os.path.join(save_path, "y_test.npy"), y_test)

    print("="*50)
    print(f"预处理完成！")
    print(f"  - 5-shot训练集规模：{len(few_shot_X_train)}")
    print(f"  - 验证集规模：{len(X_val)}")
    print(f"  - 测试集规模：{len(X_test)}")
    print(f"数据已保存至: {save_path}")
    print("="*50)
    return save_path

# ===================== 数据集加载类 =====================
class FaultDataset(Dataset):
    def __init__(self, data_path, mode="train"):
        if mode == "train":
            self.X = np.load(os.path.join(data_path, "few_shot_X_train.npy"))
            self.y = np.load(os.path.join(data_path, "few_shot_y_train.npy"))
        elif mode == "val":
            self.X = np.load(os.path.join(data_path, "X_val.npy"))
            self.y = np.load(os.path.join(data_path, "y_val.npy"))
        else:
            self.X = np.load(os.path.join(data_path, "X_test.npy"))
            self.y = np.load(os.path.join(data_path, "y_test.npy"))
        self.X = np.expand_dims(self.X, axis=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 为了演示能跑通，这里生成一些模拟数据
    # 实际使用时，请注释掉下面的模拟数据生成代码
    # ----------------------------------------------------
    print("正在生成模拟数据用于演示...")
    demo_root = "./raw_dataset"
    os.makedirs(demo_root, exist_ok=True)

    # 创建一个简单的模拟normal类别数据
    normal_dir = os.path.join(demo_root, "normal")
    os.makedirs(normal_dir, exist_ok=True)

    # 生成5个类别的模拟mat文件
    for cls_id, cls_name in enumerate(["normal", "bearing_inner", "bearing_outer", "gear_wear", "rotor_unbalance"]):
        cls_dir = os.path.join(demo_root, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        # 生成模拟信号
        np.random.seed(cls_id)
        for i in range(5): # 每个类别5个文件
            t = np.linspace(0, 10, 12000 * 10)
            signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))
            # 保存为npy模拟mat，简化演示
            np.save(os.path.join(cls_dir, f"sim_{i}.npy"), signal)

    # 重写load_raw_data用于演示
    def demo_load_raw_data(data_root):
        data_list = []
        label_list = []
        class_folders = {0: "normal", 1: "bearing_inner", 2: "bearing_outer", 3: "gear_wear", 4: "rotor_unbalance"}
        for label, folder_name in class_folders.items():
            folder_path = os.path.join(data_root, folder_name)
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".npy"):
                    raw_signal = np.load(os.path.join(folder_path, file_name))
                    data_list.append(raw_signal)
                    label_list.append(label)
        return data_list, label_list

    load_raw_data = demo_load_raw_data
    # ----------------------------------------------------

    DATA_ROOT = "./raw_dataset"
    processed_path = preprocess_pipeline(DATA_ROOT)
