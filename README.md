# -# 基于双流扩散增强的小样本工业故障诊断数据增强（DSDM）
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📝 项目简介
本项目为论文《基于双流扩散增强的小样本工业故障诊断数据增强及效果验证》的官方代码实现，**100%还原论文核心算法、实验流程与性能指标**。

针对工业场景中故障样本采集难、成本高导致的小样本诊断性能瓶颈，本项目提出**双流扩散增强模型（Dual-Stream Diffusion Model, DSDM）**：
- 特征流通过多尺度一维卷积提取振动信号的时频域多尺度特征
- 语义流编码故障类别语义信息，为生成过程提供类别约束
- 交叉注意力机制实现特征与语义的深度融合，解决生成样本类别混淆问题
- 余弦β调度策略优化噪声注入过程，提升生成样本的保真度与多样性

在5-way 5-shot极端小样本场景下，本方法可将故障诊断分类准确率提升18.7个百分点，显著优于传统噪声增强、GAN、单流扩散等主流数据增强方案。

## 📄 论文信息
- **论文标题**：基于双流扩散增强的小样本工业故障诊断数据增强及效果验证
- **作者**：陈永鑫
- **单位**：滇西科技师范学院 信息学院 智能科学与技术专业
- **指导教师**：何星宇
- **完成时间**：2026年4月

## 📋 环境要求
### 硬件环境（与论文完全对齐）
| 硬件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU 8GB显存 | NVIDIA RTX 3090 24GB显存 |
| CPU | Intel i5 8核及以上 | Intel i9-12900K |
| 运行内存 | 16GB | 64GB |

### 软件环境
- 操作系统：Windows/Linux/macOS
- 编程语言：Python 3.10+
- 深度学习框架：PyTorch 2.0+

### 依赖安装
```bash
# 安装PyTorch（根据自身CUDA版本选择对应命令，以下为CUDA 11.8版本）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy scipy matplotlib pandas scikit-learn pywt tqdm seaborn
```

## 📊 数据集说明
本项目采用工业故障诊断领域**公开标准数据集**，无私有数据，完全复现论文实验设定。

### 数据集获取
| 数据集名称 | 官方下载地址 | 备用镜像 | 覆盖故障类型 |
|------------|--------------|----------|--------------|
| PHM 2012 轴承故障数据集 | [PHM Society官网](https://www.phmsociety.org/events/conference/phm/12/data-challenge) | [GitHub镜像](https://github.com/wkzs111/phm-2012-bearing-dataset) | 正常状态、轴承内圈故障、轴承外圈故障 |
| 东南大学(SEU)齿轮箱故障数据集 | [SEU官网](https://seu11eeac.github.io/dataset/) | [GitHub镜像](https://github.com/cathysiyu/Mechanical-datasets) | 正常状态、齿轮磨损、转子不平衡 |

### 数据集目录结构
下载数据集后，按以下结构存放原始数据：
```
raw_dataset/
├── normal/              # 正常运行状态样本
├── bearing_inner/       # 轴承内圈故障样本
├── bearing_outer/       # 轴承外圈故障样本
├── gear_wear/           # 齿轮磨损故障样本
└── rotor_unbalance/     # 转子不平衡故障样本
```
每个子文件夹内存放对应类别的`.mat`格式原始振动数据。

## 🚀 快速开始
### 1. 克隆项目
```bash
git clone <项目仓库地址>
cd DSDM-Fault-Diagnosis
```

### 2. 数据准备
按照上述数据集目录结构，将下载的公开数据集原始文件放入对应文件夹。

### 3. 数据预处理
运行`preprocess.py`，完成Z-score归一化、Morlet小波时频变换、50%重叠采样、5-shot小样本数据集构建，完全对齐论文4.2节预处理流程。
```bash
python preprocess.py
```
运行完成后，预处理后的数据集将保存至`./processed_data`文件夹。

### 4. 模型训练
运行`train.py`，训练DSDM双流扩散增强模型，完全对齐论文3.6节的训练策略与参数配置。
```bash
python train.py
```
训练过程中自动保存最优模型权重至`./dsdm_model/dsdm_best.pth`，最终模型保存至`./dsdm_model/dsdm_final.pth`。

### 5. 增强样本生成
运行`inference.py`，加载训练好的模型生成故障增强样本，扩充小样本训练集。
```bash
python inference.py
```
生成的增强样本与标签将保存至`./processed_data/augmented_X.npy`与`./processed_data/augmented_y.npy`。

### 6. 故障诊断分类评估
运行`evaluate.py`，使用1D-CNN与SVM分类器，评估DSDM数据增强后的故障诊断性能，复现论文5.3节对比实验。
```bash
python evaluate.py
```
自动输出无增强基准与DSDM增强后的分类准确率、Macro-F1值等核心指标。

### 7. 消融实验
运行`ablation.py`，完成论文5.4节的模块消融实验，验证语义流、交叉注意力、余弦调度三大核心模块的性能贡献。
```bash
python ablation.py
```

### 8. 结果可视化
运行`visualization.py`，生成论文5.5节的核心可视化结果，包括t-SNE特征分布、PSD功率谱对比、不同方法准确率柱状图。
```bash
python visualization.py
```
生成的图片将自动保存至项目根目录。

## 📁 文件结构
```
DSDM-Fault-Diagnosis/
├── preprocess.py          # 数据预处理全流程代码
├── dsdm_model.py          # DSDM模型核心实现（特征流、语义流、交叉注意力、扩散模块）
├── train.py               # 模型训练代码
├── inference.py           # 增强样本生成推理代码
├── evaluate.py            # 故障诊断分类器与性能评估代码
├── ablation.py            # 消融实验代码
├── visualization.py       # 实验结果可视化代码
├── raw_dataset/           # 原始数据集存放目录
├── processed_data/        # 预处理后数据集与增强样本存放目录
├── dsdm_model/            # 模型权重保存目录
└── README.md              # 项目说明文档
```

## 📈 实验结果
### 核心性能指标（5-way 5-shot场景）
| 数据增强方案 | 1D-CNN分类准确率 | 较基准提升 | Macro-F1值 |
|--------------|------------------|------------|------------|
| 无增强(基准) | 73.6%            | -          | 69.5%      |
| 传统时域噪声增强 | 80.8%        | ↑7.2%      | 76.1%      |
| GAN数据增强 | 86.1%            | ↑12.5%     | 81.3%      |
| 单流扩散模型 | 83.9%            | ↑10.3%     | 79.2%      |
| 本文DSDM模型 | 92.3%            | ↑18.7%     | 89.6%      |

### 消融实验结果
| 模型配置 | 分类准确率 | 与完整DSDM相比变化 |
|----------|------------|--------------------|
| 完整DSDM（双流+交叉注意力+余弦调度） | 92.3% | — |
| 移除语义流模块 | 85.9% | ↓6.4% |
| 移除交叉注意力模块 | 83.1% | ↓9.2% |
| 余弦调度→线性调度 | 89.2% | ↓3.1% |

## ❓ 常见问题（FAQ）
1. **显存不足怎么办？**
   - 减小`train.py`中`batch_size`至16/8；
   - 降低`preprocess.py`中`WAVELET_SCALE`至128，减小时频图尺寸。

2. **数据集加载失败？**
   - 检查`.mat`文件的键名，不同数据集的振动信号变量名可能不同，在`preprocess.py`的`signal_keys`中补充对应键名即可。

3. **模型不收敛？**
   - 检查预处理后的信号是否完成归一化；
   - 调整学习率至5e-5，增大训练轮次至400轮；
   - 确认5-shot训练集的类别分布均衡。

4. **生成样本质量不佳？**
   - 加载训练过程中保存的最优模型`dsdm_best.pth`，而非最终模型；
   - 增大训练轮次，确保模型充分收敛。

## 📌 引用说明
如果本项目对您的研究有帮助，请引用本论文：
```
陈永鑫. 基于双流扩散增强的小样本工业故障诊断数据增强及效果验证[D]. 滇西科技师范学院, 2026.
```

## ⚠️ 免责声明
本项目仅用于学术研究，请勿用于商业用途。使用公开数据集时请遵守对应数据集的开源协议与使用规范。
