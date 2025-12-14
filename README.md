# 建筑工地安全PPE检测系统

基于YOLO的深度学习项目，用于检测建筑工地上的个人防护装备(PPE)和安全违规行为。

## 项目概述

本项目使用YOLOv8检测和分类建筑工地上的各种安全相关物体，包括：
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

## 主要特性

- 针对Windows环境优化的快速YOLO训练管道
- 自动环境初始化和依赖管理
- CUDA支持，可自动降级到CPU
- 数据增强技术（Mixup、CutMix、Mosaic、随机擦除）
- 完善的日志记录和错误处理

## 系统要求

- Python 3.8+
- PyTorch（可选CUDA支持）
- NumPy
- Ultralytics YOLOv8

## 数据集

本项目使用来自 Roboflow Universe 的建筑工地安全数据集。

### 获取数据集

访问以下链接下载数据集：
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

### 数据集结构

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

### 配置数据集路径

将下载的数据集解压到项目根目录，或在 `main.py` 中的 `CFG` 类中更新以下配置：
```python
DATA_YAML_PATH = 'data.yaml'           # 数据集配置文件路径
CSS_DATA_PATH = 'css-data'             # 数据集目录路径
BASE_MODEL = 'yolov8n.pt'              # 基础模型路径
OUTPUT_DIR = 'runs'                    # 输出目录路径
```

所有路径都是相对于项目根目录的相对路径。

## 安装步骤

1. 克隆仓库：
```bash
git clone <仓库地址>
cd YOLO_Retraining
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置说明

编辑 `main.py` 中的 `CFG` 类来自定义参数：
- `EPOCHS`: 训练轮数（默认：3）
- `BATCH_SIZE`: 批处理大小（默认：220）
- `BASE_MODEL`: 基础YOLO模型路径
- `DATA_YAML_PATH`: 数据配置文件路径
- `DEVICE`: GPU设备ID或'cpu'

## 使用方法

运行训练脚本：
```bash
python main.py
```

脚本将执行以下步骤：
1. 初始化环境并验证依赖
2. 加载预训练的YOLO模型
3. 在 `data.yaml` 指定的数据集上进行训练
4. 保存训练后的模型权重

## 项目结构

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

## 输出结果

训练结果保存在 `runs/` 目录中，包括：
- 模型权重文件（`best.pt`）
- 训练指标和曲线图
- 混淆矩阵
- 验证结果

## 注意事项

- 项目包含针对Windows的DLL和NumPy兼容性优化
- 自动管理CUDA内存以防止内存溢出
- 启用早停机制，耐心值为10个epoch
