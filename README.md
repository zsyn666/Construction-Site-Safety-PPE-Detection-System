# 建筑工地安全PPE检测系统
# Construction Site Safety PPE Detection System

一个高效的YOLOv8模型训练脚本，专门用于训练建筑工地安全防护装备(PPE)检测模型。

An efficient YOLOv8 model training script specifically designed for training Personal Protective Equipment (PPE) detection models on construction sites.

## 项目概述 | Overview

本项目提供了一个完整的模型训练管道，用于训练YOLOv8模型以检测和分类建筑工地上的各种安全相关物体，包括：
- 安全帽
- 口罩
- 未戴安全帽
- 未戴口罩
- 未穿安全背心
- 人员
- 安全锥
- 安全背心
- 机械设备
- 车辆

## 主要特性 | Features

- **完整的训练管道** - 从环境初始化到模型保存的一站式解决方案
- **Windows优化** - 针对Windows环境的特殊优化，解决DLL和NumPy兼容性问题
- **灵活的设备支持** - 自动检测CUDA可用性，支持GPU加速或CPU训练
- **智能内存管理** - 自动管理CUDA内存，防止内存溢出错误
- **完善的日志记录** - 详细的训练过程日志和错误提示
- **易于配置** - 所有参数集中在CFG类中，便于快速调整

## 系统要求 | Requirements

- Python 3.8+
- PyTorch（可选CUDA支持）
- NumPy
- Ultralytics YOLOv8

## 数据集 | Dataset

本项目使用来自 Roboflow Universe 的建筑工地安全数据集。

### 获取数据集 | Download Dataset

访问以下链接下载数据集：
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

### 数据集结构 | Dataset Structure

下载后的数据集应包含以下结构：
```
css-data/
├── train/
│   ├── images/     # 训练集图像
│   └── labels/     # 训练集标签（YOLO格式）
├── valid/
│   ├── images/     # 验证集图像
│   └── labels/     # 验证集标签（YOLO格式）
├── test/
│   ├── images/     # 测试集图像
│   └── labels/     # 测试集标签（YOLO格式）
└── data.yaml       # 数据集配置文件
```

### 配置数据集路径 | Configure Dataset Path

将下载的数据集解压到项目根目录，或在 `main.py` 中的 `CFG` 类中更新以下配置：
```python
DATA_YAML_PATH = 'data.yaml'           # 数据集配置文件路径
CSS_DATA_PATH = 'css-data'             # 数据集目录路径
BASE_MODEL = 'yolov8n.pt'              # 基础模型路径
OUTPUT_DIR = 'runs'                    # 输出目录路径
```

所有路径都是相对于项目根目录的相对路径。

## 安装步骤 | Installation

1. 克隆仓库：
```bash
git clone <仓库地址>
cd YOLO_Retraining
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置说明 | Configuration

### 训练参数 | Training Parameters

编辑 `main.py` 中的 `CFG` 类来自定义训练参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `EPOCHS` | 训练轮数 | 3 |
| `BATCH_SIZE` | 批处理大小 | 220 |
| `BASE_MODEL` | 基础YOLO模型 | yolov8n.pt |
| `DATA_YAML_PATH` | 数据配置文件路径 | data.yaml |
| `CSS_DATA_PATH` | 数据集目录路径 | css-data |
| `OUTPUT_DIR` | 输出目录 | runs |
| `DEVICE` | 训练设备 | 0 (GPU) 或 'cpu' |
| `CLEAN_PREVIOUS_RUNS` | 清理之前的训练结果 | True |

### 高级配置 | Advanced Configuration

训练脚本还支持以下高级参数（在 `train_args` 字典中）：
- `imgsz`: 输入图像大小（默认：320）
- `patience`: 早停耐心值（默认：10）
- `lr0`: 初始学习率（默认：0.01）
- `momentum`: 动量值（默认：0.937）
- `optimizer`: 优化器类型（默认：SGD）

## 使用方法 | Usage

### 快速开始 | Quick Start

1. 准备数据集（参考数据集部分）
2. 根据需要修改 `main.py` 中的 `CFG` 类配置
3. 运行训练脚本：

```bash
python main.py
```

### 训练流程 | Training Process

脚本将自动执行以下步骤：

1. **环境初始化** - 检查并初始化Python环境，验证NumPy和PyTorch
2. **依赖验证** - 确保所有必需的库都已正确安装
3. **模型加载** - 加载预训练的YOLOv8基础模型
4. **数据验证** - 检查数据集配置文件和路径
5. **模型训练** - 在指定数据集上进行模型训练
6. **权重保存** - 保存最佳模型权重到指定目录

## 项目结构 | Project Structure

```
.
├── main.py                 # 主训练脚本
├── data.yaml              # 数据集配置文件
├── requirements.txt       # Python依赖列表
├── css-data/              # 数据集目录
│   ├── train/            # 训练集（图像和标签）
│   ├── valid/            # 验证集（图像和标签）
│   └── test/             # 测试集（图像）
└── runs/                 # 训练输出和结果
```

## 输出结果 | Output

训练完成后，结果保存在 `runs/{EXP_NAME}/` 目录中，包括：

```
runs/css_ppe_fast/
├── weights/
│   ├── best.pt                    # 最佳模型权重
│   ├── last.pt                    # 最后一个epoch的权重
│   └── SafetyHelmetWearing.pt     # 自定义命名的权重副本
├── results.csv                    # 训练指标CSV文件
├── confusion_matrix.png           # 混淆矩阵
├── confusion_matrix_normalized.png # 归一化混淆矩阵
├── P_curve.png                    # 精确率曲线
├── R_curve.png                    # 召回率曲线
├── F1_curve.png                   # F1分数曲线
├── PR_curve.png                   # P-R曲线
└── val_batch*.jpg                 # 验证集可视化结果
```

## 故障排除 | Troubleshooting

### GPU内存不足
如果遇到 "out of memory" 错误：
- 减少 `BATCH_SIZE` 的值
- 使用更小的基础模型（如 `yolov8n.pt` 而不是 `yolov8l.pt`）
- 减少 `imgsz` 的值

### 导入错误
如果遇到库导入错误：
```bash
pip install --upgrade torch torchvision torchaudio
pip install --upgrade ultralytics
pip install --upgrade numpy
```

### Windows DLL错误
脚本已包含Windows兼容性优化，但如果仍有问题：
- 确保使用最新的Python版本（3.9+）
- 重新安装NumPy：`pip uninstall numpy && pip install numpy`

## 注意事项 | Notes

- 项目包含针对Windows的DLL和NumPy兼容性优化
- 自动管理CUDA内存以防止内存溢出
- 启用早停机制，耐心值为10个epoch
- 所有路径配置都使用相对路径，便于跨平台使用
- 训练过程中会自动清理之前的训练结果（如果启用 `CLEAN_PREVIOUS_RUNS`）
