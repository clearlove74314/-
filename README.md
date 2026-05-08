# 基于双流扩散增强的小样本工业故障诊断数据增强及效果验证
## DSDM: Dual-Stream Diffusion Enhancement Model for Few-Shot Industrial Fault Diagnosis

本项目为论文《基于双流扩散增强的小样本工业故障诊断数据增强及效果验证》的官方代码实现，针对工业故障诊断场景中故障样本稀缺、小样本下分类精度不足的痛点，构建了双流扩散增强模型（DSDM），实现了高保真的故障振动信号数据增强，配套完整的实验流程、对照实验、消融实验与可视化分析全链路代码。

核心实验流程遵循**「训练池预训练—5-shot 微调—增强样本生成—分类器性能验证」**的闭环设计，可完整复现论文中的全部实验结果。

---

## 核心特性
- ✅ 工业故障振动信号全流程预处理，**时序块隔离的无泄露数据划分**，从根源避免训练/验证/测试集数据交叉
- ✅ 标准 5-way 5-shot 小样本故障诊断任务构建，支持多随机种子重复实验，保证结果可复现
- ✅ 完整的双流扩散增强模型 DSDM 实现，包含故障特征流、故障语义流、交叉注意力交互、条件扩散解码与频率保真损失模块
- ✅ 多组对照实验支持：Add 融合、Concat 融合、Linear 噪声调度等变体模型一键运行
- ✅ 双分类器性能验证：1-D CNN 深度分类器 + 传统 SVM 分类器，全面验证增强样本泛化性
- ✅ 多维度生成样本质量评价：PSD-MSE、MMD、DTW、FD、KS 检验等量化指标全覆盖
- ✅ 完整的消融实验链路，可一键验证各模块的性能贡献
- ✅ 论文级可视化支持：时域波形、频谱对比、t-SNE 分布、混淆矩阵、注意力权重热力图等

---

## 环境配置
### 1. Python 环境
推荐使用 Python 3.10，通过 conda 创建虚拟环境：
```bash
conda create -n dsdm_fault python=3.10 -y
conda activate dsdm_fault
```

### 2. 依赖安装
方式一：一键安装全部依赖
```bash
pip install torch torchvision torchaudio numpy scipy pandas scikit-learn matplotlib seaborn pywavelets tqdm einops
```

方式二：通过 requirements.txt 安装
```bash
pip install -r requirements.txt
```

`requirements.txt` 内容如下：
```text
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pywavelets>=1.4.0
tqdm>=4.65.0
einops>=0.6.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
```

---

## 项目目录结构
```text
DSDM_FewShot_FaultDiagnosis/
├── README.md                           # 项目说明文档
├── requirements.txt                    # 项目依赖
├── config/                             # 配置文件目录
│   ├── dsdm.yaml                       # DSDM 模型核心配置
│   ├── classifier.yaml                 # 分类器配置
│   └── ablation.yaml                   # 消融实验配置
├── data/                               # 数据目录
│   ├── raw/                            # 原始振动数据，按类别分文件夹存放
│   ├── processed/                      # 预处理后的数据、划分好的数据集与支持集
│   └── generated/                      # 各模型生成的增强样本
├── src/                                # 核心代码目录
│   ├── data/                           # 数据处理相关代码
│   ├── models/                         # DSDM 模型与分类器实现
│   ├── losses/                         # 损失函数定义
│   ├── train/                          # 模型预训练、微调、样本生成与分类器训练
│   ├── eval/                           # 分类性能、生成质量与统计检验代码
│   ├── visualize/                      # 论文全系列可视化绘图代码
│   └── utils/                          # 工具函数（随机种子、日志、指标计算等）
├── experiments/                        # 批量实验脚本
│   ├── main_dsdm.sh                    # 主实验脚本
│   ├── compare_fusion.sh               # 融合方式对照实验
│   ├── compare_scheduler.sh            # 噪声调度对照实验
│   ├── ablation.sh                     # 消融实验脚本
│   └── repeat_5seeds.sh                # 5次随机种子重复实验一键运行
├── outputs/                            # 实验输出目录
│   ├── checkpoints/                    # 模型权重文件
│   ├── logs/                           # 训练与测试日志
│   ├── metrics/                        # 量化指标计算结果
│   └── figures/                        # 可视化绘图结果
└── results/                            # 最终实验结果汇总
    ├── classification_results.csv      # 分类性能结果汇总
    ├── generation_quality.csv          # 生成质量指标汇总
    ├── ablation_results.csv            # 消融实验结果
    └── statistical_test.csv            # 统计显著性检验结果
```

---

## 数据准备
### 1. 数据类别与格式
本实验针对5类典型的设备运行状态识别任务，原始数据需按类别放入对应文件夹，结构如下：
```text
data/raw/
├── normal/             # 0 正常状态
├── inner_race/         # 1 轴承内圈故障
├── outer_race/         # 2 轴承外圈故障
├── gear_wear/          # 3 齿轮磨损
└── rotor_imbalance/    # 4 转子不平衡
```
支持的原始数据格式：`.csv`、`.txt`、`.npy`、`.mat`，若使用`.mat`格式，需在预处理代码中指定对应变量名。

### 2. 核心数据参数
论文实验采用的核心参数如下，可根据自有数据调整：
| 参数 | 数值 | 说明 |
|---|---|---|
| 采样频率 | 12000 Hz | 振动信号采样频率 |
| 转速 | 1500 r/min | 设备运行转速 |
| 转频 | 25 Hz | 设备旋转频率 |
| 窗口长度 | 2048 | 单样本采样点长度 |
| 滑动步长 | 1024 | 滑窗采样步长 |
| 窗口重叠率 | 50% | 相邻样本重叠比例 |
| 训练/验证/测试比例 | 8:1:1 | 时序块划分比例 |
| 块隔离间隔 | 2048 采样点 | 数据集间防泄露隔离带 |

---

## 完整实验复现流程
### 1. 数据预处理与泄露校验
为彻底避免数据泄露，本项目采用**先时序分块、后块内滑窗**的划分方式，而非传统的先滑窗后随机划分，同时在数据集间设置隔离间隔，保证无采样点交叉。

#### 1.1 数据预处理
```bash
python src/data/preprocess.py \
    --raw_dir data/raw \
    --save_dir data/processed \
    --fs 12000 \
    --window_size 2048 \
    --stride 1024 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --gap 2048
```
运行完成后，会在 `data/processed/` 下生成划分好的训练集、验证集、测试集，以及对应样本的原始采样点索引记录。

#### 1.2 数据泄露检查
校验训练/验证/测试集之间无采样点重叠，保证实验严谨性：
```bash
python src/data/leakage_check.py \
    --index_dir data/processed/index_record
```
**期望输出**：三组数据集交集均为0，提示无采样点泄露。

#### 1.3 故障特征频率计算
计算对应设备与故障类型的理论特征频率，用于后续信号分析与频率保真约束：
```bash
python src/data/frequency_analysis.py \
    --data_dir data/processed/train \
    --fs 12000 \
    --rpm 1500 \
    --num_balls 8 \
    --diameter_ratio 0.206 \
    --contact_angle 0 \
    --gear_teeth 32
```
运行后会输出转频、轴承内外圈故障频率、齿轮啮合频率等理论值，与论文中参数一致。

### 2. 5-way 5-shot 小样本任务构建
为保证实验可复现，基于5组固定随机种子，为每类故障构建5个支持样本，形成标准小样本任务：
```bash
python src/data/build_fewshot_task.py \
    --train_dir data/processed/train \
    --save_dir data/processed/support_set \
    --num_classes 5 \
    --shot 5 \
    --seeds 2024 2025 2026 2027 2028
```
运行后会在 `data/processed/support_set/` 下生成对应随机种子的支持集，包含样本、标签与对应索引记录。

### 3. DSDM 模型训练
#### 3.1 预训练
预训练阶段仅使用训练池数据，学习工业振动信号的条件生成先验，验证集仅用于早停与超参数选择，不参与模型权重更新：
```bash
python src/train/pretrain_dsdm.py \
    --config config/dsdm.yaml \
    --train_dir data/processed/train \
    --val_dir data/processed/val \
    --save_dir outputs/checkpoints/dsdm_pretrain \
    --seed 2024
```
预训练完成后，最优模型权重会保存至 `outputs/checkpoints/dsdm_pretrain/best_model.pth`。

#### 3.2 5-shot 微调
针对小样本场景，基于预训练权重，使用每类5个支持样本进行微调，仅更新语义嵌入、条件归一化、交叉注意力与输出投影层参数，避免过拟合：
```bash
# 单种子微调示例（seed=2024）
python src/train/finetune_dsdm.py \
    --config config/dsdm.yaml \
    --pretrained_ckpt outputs/checkpoints/dsdm_pretrain/best_model.pth \
    --support_dir data/processed/support_set/seed_2024 \
    --save_dir outputs/checkpoints/dsdm_finetune/seed_2024 \
    --seed 2024
```
5组随机种子的完整微调可直接运行批量脚本：
```bash
bash experiments/repeat_5seeds.sh
```

### 4. 增强样本生成
基于微调后的DSDM模型，为每类故障生成200个增强样本，用于后续分类器训练：
```bash
# 单种子生成示例
python src/train/generate_samples.py \
    --config config/dsdm.yaml \
    --ckpt outputs/checkpoints/dsdm_finetune/seed_2024/best_model.pth \
    --support_dir data/processed/support_set/seed_2024 \
    --save_dir data/generated/dsdm/seed_2024 \
    --samples_per_class 200 \
    --seed 2024
```
批量生成可直接使用 `experiments/repeat_5seeds.sh` 脚本。

### 5. 分类器性能验证
使用「真实5-shot支持样本 + DSDM生成增强样本」训练分类器，在独立测试集上验证性能，测试集全程不参与任何训练过程。

#### 5.1 1-D CNN 分类器
```bash
# 单种子示例
python src/train/train_cnn.py \
    --config config/classifier.yaml \
    --support_dir data/processed/support_set/seed_2024 \
    --generated_dir data/generated/dsdm/seed_2024 \
    --test_dir data/processed/test \
    --save_dir outputs/logs/cnn/dsdm/seed_2024 \
    --seed 2024
```

#### 5.2 SVM 分类器
补充验证增强样本对传统分类器的泛化性：
```bash
# 单种子示例
python src/train/train_svm.py \
    --support_dir data/processed/support_set/seed_2024 \
    --generated_dir data/generated/dsdm/seed_2024 \
    --test_dir data/processed/test \
    --save_dir outputs/logs/svm/dsdm/seed_2024 \
    --seed 2024
```

### 6. 对照实验
论文包含3组核心对照实验，用于验证DSDM各模块的有效性，所有对照实验均使用与主实验完全一致的支持集、生成样本数量与分类器结构。

| 对照模型 | 改动说明 |
|---|---|
| DSDM-Add | 将交叉注意力替换为逐元素相加融合 |
| DSDM-Concat | 将交叉注意力替换为特征拼接+MLP融合 |
| DSDM-Linear | 保留双流与交叉注意力，将余弦噪声调度替换为线性调度 |

一键运行全部对照实验：
```bash
bash experiments/compare_fusion.sh
bash experiments/compare_scheduler.sh
```

### 7. 消融实验
通过逐步添加模块，验证各组件对模型性能的贡献，完整消融实验设置如下：
| 编号 | 模型设置 |
|---|---|
| 1 | 单流 DDPM + 线性调度 |
| 2 | 单流 DDPM + 余弦调度 |
| 3 | 加入多尺度特征流 |
| 4 | 加入语义流 |
| 5 | 加入交叉注意力 |
| 6 | 加入频率保真损失（完整DSDM） |

一键运行消融实验：
```bash
bash experiments/ablation.sh
```

### 8. 生成质量量化评价
从时域、频域、分布一致性等多维度，评价生成样本的质量，核心指标包括PSD-MSE、MMD、DTW、FD、KS检验通过率：
```bash
# DSDM 主模型生成质量评价
python src/eval/evaluate_generation.py \
    --real_dir data/processed/train \
    --generated_dir data/generated/dsdm/seed_2024 \
    --save_dir outputs/metrics/generation/dsdm/seed_2024 \
    --fs 12000
```
对照模型的评价可通过修改 `--generated_dir` 参数批量运行。

### 9. 统计显著性检验
基于5次重复实验结果，通过配对t检验验证DSDM相比对照组的性能提升是否具有统计显著性：
```bash
python src/eval/statistical_test.py \
    --result_csv results/classification_results.csv \
    --target_method DSDM \
    --compare_methods DSDM-Add DSDM-Concat DSDM-Linear \
    --metric accuracy \
    --save_path results/statistical_test.csv
```

### 10. 论文级可视化
提供论文中全部图表的一键生成代码，所有图片均保存至 `outputs/figures/` 目录：
```bash
# 时域波形图
python src/visualize/plot_waveform.py \
    --data_dir data/processed/train \
    --save_path outputs/figures/fig_4_3_waveform.png \
    --num_classes 5

# 频谱图
python src/visualize/plot_spectrum.py \
    --data_dir data/processed/train \
    --fs 12000 \
    --save_path outputs/figures/fig_4_4_spectrum.png

# 分类准确率柱状图
python src/visualize/plot_results.py \
    --result_csv results/classification_results.csv \
    --metric accuracy \
    --save_path outputs/figures/fig_5_1_accuracy.png

# 消融实验结果图
python src/visualize/plot_results.py \
    --result_csv results/ablation_results.csv \
    --metric accuracy \
    --save_path outputs/figures/fig_5_2_ablation.png

# 真实与生成样本频谱对比图
python src/visualize/plot_spectrum.py \
    --real_dir data/processed/train \
    --generated_dir data/generated/dsdm/seed_2024 \
    --fs 12000 \
    --compare_real_generated \
    --save_path outputs/figures/fig_5_3_real_generated_spectrum.png

# t-SNE 分布可视化
python src/visualize/plot_tsne.py \
    --real_dir data/processed/train \
    --generated_dir data/generated/dsdm/seed_2024 \
    --save_path outputs/figures/fig_5_4_tsne.png

# 注意力权重热力图
python src/visualize/plot_attention.py \
    --ckpt outputs/checkpoints/dsdm_finetune/seed_2024/best_model.pth \
    --support_dir data/processed/support_set/seed_2024 \
    --save_path outputs/figures/fig_5_5_attention.png

# 混淆矩阵
python src/eval/confusion_matrix.py \
    --pred_path outputs/logs/cnn/dsdm/seed_2024/predictions.npy \
    --label_path outputs/logs/cnn/dsdm/seed_2024/labels.npy \
    --save_path outputs/figures/fig_5_6_confusion_matrix.png
```

---

## 核心实验结果
运行完整实验后，可复现论文中的核心结果，如下所示：

### 1. 1-D CNN 分类性能结果
| 方法 | Accuracy/% | Macro-F1/% | AUC | PSD-MSE | MMD | p 值 |
|---|---:|---:|---:|---:|---:|---:|
| DSDM-Add | 86.5±1.7 | 83.4±1.9 | 0.925±0.013 | 0.019 | 0.061 | 0.004 |
| DSDM-Concat | 88.7±1.5 | 85.9±1.8 | 0.941±0.011 | 0.016 | 0.052 | 0.018 |
| DSDM-Linear | 89.2±1.4 | 86.8±1.6 | 0.947±0.010 | 0.021 | 0.055 | 0.026 |
| DSDM | 92.3±1.2 | 89.6±1.5 | 0.963±0.008 | 0.012 | 0.043 | — |

### 2. SVM 分类性能结果
| 方法 | Accuracy/% | Macro-F1/% |
|---|---:|---:|
| DSDM-Add | 82.4±2.1 | 79.5±2.3 |
| DSDM-Concat | 85.1±1.8 | 82.7±2.0 |
| DSDM-Linear | 85.6±1.7 | 83.2±1.9 |
| DSDM | 88.7±1.5 | 86.1±1.7 |

### 3. 消融实验结果
| 模型设置 | Accuracy/% | Macro-F1/% | PSD-MSE | 准确率提升 | p 值 |
|---|---:|---:|---:|---:|---:|
| 单流 DDPM + 线性调度 | 82.6±2.2 | 79.4±2.5 | 0.026 | — | — |
| 单流 DDPM + 余弦调度 | 85.4±1.9 | 82.1±2.2 | 0.021 | +2.8 | 0.041 |
| 加入多尺度特征流 | 88.4±1.6 | 85.3±1.8 | 0.017 | +3.0 | 0.033 |
| 加入语义流 | 89.8±1.5 | 86.9±1.7 | 0.015 | +1.4 | 0.047 |
| 加入交叉注意力 | 91.5±1.3 | 88.7±1.6 | 0.013 | +1.7 | 0.029 |
| 完整 DSDM | 92.3±1.2 | 89.6±1.5 | 0.012 | +0.8 | 0.038 |

### 4. 生成样本质量评价结果
| 方法 | PSD-MSE | MMD | DTW | FD | KS 检验通过率/% |
|---|---:|---:|---:|---:|---:|
| DSDM-Add | 0.019 | 0.061 | 0.284 | 13.72 | 76.0 |
| DSDM-Concat | 0.016 | 0.052 | 0.241 | 10.46 | 82.0 |
| DSDM-Linear | 0.021 | 0.055 | 0.266 | 11.38 | 80.0 |
| DSDM | 0.012 | 0.043 | 0.198 | 7.84 | 92.0 |

---

## 注意事项
1. 测试集全程不得参与DSDM预训练、微调、增强样本筛选或分类器训练的任何环节。
2. 数据归一化参数仅能通过训练集计算，不得使用验证集或测试集的统计量。
3. 必须严格遵循「先时序分块、后块内滑窗」的划分方式，禁止先滑窗后随机划分，避免数据泄露。
4. 每次重复实验必须完整保存随机种子、支持集索引与生成样本索引，保证结果可复现。
5. 所有对照实验与消融实验，必须使用与主实验完全一致的支持集、生成样本数量与分类器结构，保证变量唯一。
6. t-SNE仅作为分布可视化的辅助手段，不得作为生成质量的唯一评价依据。
7. 论文中报告的均值与标准差，均来自5次独立随机种子的重复实验，单次实验结果可能存在小幅波动。
8. 若更换自有数据集或调整数据划分方式，需重新运行全部实验并更新对应结果。

---

## 许可证
本项目仅用于论文实验复现与学术研究用途。若使用公开工业故障数据集，请严格遵守原数据集发布方的使用规范与许可协议。若基于本项目进行扩展研究或二次开发，请在相关论文、报告或项目中注明来源与引用。
