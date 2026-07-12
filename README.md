# 眼底多模态血管分割与出血点一体化检测系统

本项目面向眼底影像分析，包含 CFP 彩色眼底血管分割、OCTA 血流图血管分割、CFP+OCTA 多模态融合血管分割、眼底出血点实例分割，以及一个用于模型推理的可视化 GUI。项目中的训练脚本、推理脚本和 GUI 均使用相对路径或脚本所在目录定位资源，移动整个 `Vascular_Imaging_Project` 文件夹后通常不需要修改代码路径。

> 说明：本项目用于科研、课程或算法实验，不应直接作为临床诊断工具使用。

## 功能概览

1. CFP 单模态血管分割

- 存储路径：`Vessel_Segmentation/CFP_Vessel_Project/UNets-master`
- 使用模型：UNet 及其变体
- 主要输入：DRIVE CFP 图像
- 主要输出：血管分割结果、评价指标、损失曲线

2. OCTA 单模态血管分割

- 存储路径：`Vessel_Segmentation/OCTA_Vessel_Project`
- 使用模型：UNet
- 主要输入：OCTA 灰度图像
- 主要输出：血管分割结果、训练损失曲线

3. 多模态融合血管分割

- 存储路径：`Vessel_Segmentation/MultiModal_Vessel_Project`
- 使用模型：EarlyFusionUNet、AttentionFusionUNet
- 主要输入：CFP 与 OCTA / 伪 OCTA 配对图像数据
- 主要输出：多模态融合血管分割结果

4. 出血点实例分割

- 存储路径：`Bleeding_Spot_detection`
- 使用模型：YOLOv8-Seg
- 主要输入：IDRiD 数据集 CFP 图像
- 主要输出：出血点二值掩码、原图叠加可视化效果图

5. GUI 可视化推理工具

- 存储路径：`Software`
- 使用模型：AttentionFusionUNet + YOLOv8-Seg
- 主要输入：单张眼底图像或批量图像文件夹
- 主要输出：血管分割掩码、出血点检测掩码

## 获取项目

仓库中的大图像和模型权重通过 Git LFS 管理。首次克隆后必须拉取 LFS 文件，否则 `.pt`、`.pth`、`.tif` 等文件可能只是文本指针，运行模型时会加载失败。

```bash
git lfs install
git clone https://github.com/JanePeng-maker/Vascular_Imaging_Project.git
cd Vascular_Imaging_Project
git lfs pull
```

如果已经普通克隆过仓库，也可以在项目根目录执行：

```bash
git lfs install
git lfs pull
```

## 环境配置

建议使用 Python 3.8 到 3.12。若有 NVIDIA GPU，可安装支持 CUDA 的 PyTorch；没有 GPU 时也能使用 CPU 运行，但训练速度会较慢。

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

安装通用依赖：

```bash
pip install torch torchvision opencv-python numpy matplotlib scipy scikit-image imageio tifffile Pillow PyYAML ultralytics
```

如果只运行 GUI，也需要安装上述依赖，因为 GUI 同时加载 PyTorch 血管分割模型和 Ultralytics YOLO 模型。

macOS 通常自带 `tkinter`。如果运行 GUI 时报 `No module named tkinter`，需要安装带 Tk 支持的 Python 版本。

## 项目结构

```text
Vascular_Imaging_Project/
├── README.md
├── Vessel_Segmentation/
│   ├── CFP_Vessel_Project/
│   │   └── UNets-master/
│   │       ├── datasets/
│   │       │   ├── training/
│   │       │   │   ├── images/
│   │       │   │   └── 1st_manual/
│   │       │   └── test/
│   │       │       ├── images/
│   │       │       └── 1st_manual/
│   │       ├── nets/
│   │       ├── result/
│   │       ├── saved_model/
│   │       ├── saved_predict/
│   │       ├── dataset.py
│   │       ├── main.py
│   │       ├── metrics.py
│   │       └── plot.py
│   │
│   ├── OCTA_Vessel_Project/
│   │   ├── train_images/
│   │   ├── train_masks/
│   │   ├── test_images/
│   │   ├── test_masks/
│   │   ├── saved_model/
│   │   ├── test_results/
│   │   ├── dataset.py
│   │   ├── main.py
│   │   └── model.py
│   │
│   └── MultiModal_Vessel_Project/
│       ├── code/
│       │   ├── dataset.py
│       │   ├── generate_pseudo_pair.py
│       │   ├── main.py
│       │   └── model.py
│       ├── data/
│       │   ├── cfp_original/
│       │   ├── octa_original/
│       │   ├── cfp_pretrained/
│       │   ├── octa_pretrained/
│       │   └── pseudo_pair/
│       ├── saved_models/
│       └── results/
│
├── Bleeding_Spot_detection/
│   ├── Originaldata/
│   ├── GT/
│   ├── dataset/
│   ├── runs/
│   ├── results/
│   ├── config.yaml
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
│
└── Software/
    ├── inference_gui.py
    ├── att_fusion_best.pth
    └── best.pt
```

## 数据集放置

### DRIVE 数据集

CFP 单模态模块和多模态伪配对生成脚本都使用 DRIVE 风格目录。

CFP 单模态模块数据集存放（替换）要求：

```text
Vessel_Segmentation/CFP_Vessel_Project/UNets-master/datasets/
├── training/
│   ├── images/
│   └── 1st_manual/
└── test/
    ├── images/
    └── 1st_manual/
```

注意：代码中的 `DATASET_ROOT` 直接指向 `UNets-master/datasets`，因此不要额外套一层 `DRIVE/`，除非同步修改 `main.py`。

### OCTA 数据集

OCTA 单模态模块存放（替换）要求：

```text
Vessel_Segmentation/OCTA_Vessel_Project/
├── train_images/
├── train_masks/
├── test_images/
└── test_masks/
```

当前 `dataset.py` 默认读取 `.png` 图像。若数据是其他格式，需要改 `dataset.py` 中的 `glob(os.path.join(..., '*.png'))`。

### IDRiD 出血点数据集

出血点模块存放（替换）要求：

```text
Bleeding_Spot_detection/
├── Originaldata/
│   ├── Training Set/
│   └── Testing Set/
└── GT/
    ├── Training Set/
    └── Testing Set/
```

`prepare_data.py` 会按图像名寻找对应掩码，掩码文件名必须形如：

```text
IDRiD_01_HE.tif
```

预处理后会生成 YOLO 格式数据：

```text
Bleeding_Spot_detection/dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## CFP 单模态血管分割

进入 CFP 模块目录：

```bash
cd Vascular_Imaging_Project/Vessel_Segmentation/CFP_Vessel_Project/UNets-master
```

默认运行训练和测试：

```bash
python3 main.py
```

默认参数在 `main.py` 中设置：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--action` | `train&test` | 训练后测试 |
| `--arch` | `UNet` | 网络结构 |
| `--epoch` | `10` | 训练轮数 |
| `--batch_size` | `1` | 批大小 |
| `--num_test` | `-1` | 测试全部图像 |

只测试已有权重：

```bash
python3 main.py --action test --model_path ./saved_model/cfp_unet_best.pth
```

指定模型结构和训练轮数：

```bash
python3 main.py --arch Attention_UNet --epoch 20
```

如果显式传入 `train&test`，不同终端可能会把 `&` 当作后台符号，建议加引号：

```bash
python3 main.py --action "train&test"
```

输出位置：

| 输出 | 路径 |
|---|---|
| 最优模型 | `saved_model/cfp_unet_best.pth` |
| 训练日志 | `result/log/` |
| 损失曲线和指标曲线 | `result/plot/` |
| 测试三联对比图 | `saved_predict/` |

`saved_predict` 保存的是输入图、预测 mask、真实 mask 的三栏对比图，不是单独的二值掩码文件。

## OCTA 单模态血管分割

进入 OCTA 模块目录：

```bash
cd Vascular_Imaging_Project/Vessel_Segmentation/OCTA_Vessel_Project
```

`main.py` 顶部有两个运行开关：

```python
RUN_TRAIN = False
RUN_TEST = True
```

如果是第一次运行且还没有 `saved_model/best_octa_unet.pth`，需要先训练：

```python
RUN_TRAIN = True
RUN_TEST = False
```

然后执行：

```bash
python3 main.py
```

训练完成后，如需测试，改为：

```python
RUN_TRAIN = False
RUN_TEST = True
```

再次执行：

```bash
python3 main.py
```

输出位置：

| 输出 | 路径 |
|---|---|
| 最优模型 | `saved_model/best_octa_unet.pth` |
| 损失曲线 | `saved_model/loss_curve.png` |
| 测试对比图 | `test_results/` |

可调参数集中在 `main.py` 顶部，包括 `N_TRAIN_SAMPLES`、`N_TEST_SAMPLES`、`IMG_SIZE`、`EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE`、`POS_WEIGHT`。

## 多模态融合血管分割

多模态模块位于：

```text
Vessel_Segmentation/MultiModal_Vessel_Project/
```

该模块支持两种模型：

| 模型 | 配置值 | 说明 |
|---|---|---|
| EarlyFusionUNet | `MODEL_TYPE = "early_fusion"` | 直接拼接 CFP 3 通道与 OCTA 1 通道 |
| AttentionFusionUNet | `MODEL_TYPE = "att_fusion"` | 双流编码器 + SE 注意力特征融合 |

### 准备单模态预训练权重

如果使用默认配置：

```python
LOAD_PRETRAINED = True
FREEZE_ENCODER = True
MODEL_TYPE = "att_fusion"
```

需要放置两个单模态权重，并使用代码中指定的文件名：

```text
Vessel_Segmentation/MultiModal_Vessel_Project/data/
├── cfp_pretrained/
│   └── cfp_unet_best.pth
└── octa_pretrained/
    └── octa_unet_best.pth
```

其中：

| 权重 | 来源 |
|---|---|
| `cfp_unet_best.pth` | CFP 单模态模块训练得到的 `saved_model/cfp_unet_best.pth` |
| `octa_unet_best.pth` | OCTA 单模态模块训练得到的 `saved_model/best_octa_unet.pth`，复制后重命名 |

如果不想加载单模态预训练权重，可在 `code/main.py` 中改为：

```python
LOAD_PRETRAINED = False
FREEZE_ENCODER = False
```

### 准备伪配对数据

`generate_pseudo_pair.py` 通过 CFP 灰度图与 OCTA 参考图做直方图匹配，生成与 CFP 同名的伪 OCTA 图像。该方法用于实验和演示，不等同于真实配准的 CFP-OCTA 数据。

脚本要求：

```text
Vessel_Segmentation/MultiModal_Vessel_Project/data/
├── cfp_original/
│   ├── training/
│   │   ├── images/
│   │   └── 1st_manual/
│   └── test/
│       ├── images/
│       └── 1st_manual/
└── octa_original/
    └── train_images/
```

生成伪配对数据：

```bash
cd Vascular_Imaging_Project/Vessel_Segmentation/MultiModal_Vessel_Project/code
python3 generate_pseudo_pair.py
```

输出位置：

```text
Vessel_Segmentation/MultiModal_Vessel_Project/data/pseudo_pair/
├── train/
│   ├── cfp/
│   ├── octa/
│   └── mask/
└── test/
    ├── cfp/
    ├── octa/
    └── mask/
```

### 训练融合模型

在 `code/main.py` 中设置：

```python
RUN_TRAIN = True
RUN_TEST = False
```

执行：

```bash
cd Vascular_Imaging_Project/Vessel_Segmentation/MultiModal_Vessel_Project/code
python3 main.py
```

输出位置：

```text
Vessel_Segmentation/MultiModal_Vessel_Project/saved_models/att_fusion_best.pth
Vessel_Segmentation/MultiModal_Vessel_Project/results/att_fusion_training_curve.png
```

### 测试融合模型

在 `code/main.py` 中设置：

```python
RUN_TRAIN = False
RUN_TEST = True
```

执行：

```bash
python3 main.py
```

测试输出：

```text
Vessel_Segmentation/MultiModal_Vessel_Project/results/att_fusion/
```

## 出血点检测

进入出血点模块：

```bash
cd Vascular_Imaging_Project/Bleeding_Spot_detection
```

### 数据预处理

如果 `dataset/` 已存在且包含 `images/train`、`images/val`、`labels/train`、`labels/val`，可跳过该步骤。否则执行：

```bash
python3 prepare_data.py
```

该脚本会读取：

```text
Originaldata/Training Set
Originaldata/Testing Set
GT/Training Set
GT/Testing Set
```

并生成：

```text
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

### 训练 YOLOv8-Seg

推荐使用项目自带训练脚本：

```bash
python3 train.py
```

`train.py` 会自动定位当前模块目录，并读取 `config.yaml`。默认参数包括：

| 参数 | 默认值 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 轮数 | `10` |
| 图像尺寸 | `640` |
| 批大小 | `8` |
| 设备 | `cpu` |
| 优化器 | `AdamW` |

如果使用 GPU，可在 `train.py` 中把：

```python
DEVICE = "cpu"
```

改为：

```python
DEVICE = 0
```

训练完成后，最优权重通常位于：

```text
Bleeding_Spot_detection/runs/segment/train/weights/best.pt
```

如果多次训练产生了嵌套目录，以终端最后打印的 `最佳权重完整路径` 为准。

### 使用命令行训练

也可以直接使用 Ultralytics 命令：

```bash
yolo task=segment mode=train data=config.yaml model=yolov8s-seg.pt epochs=100 imgsz=640
```

执行该命令时必须位于 `Bleeding_Spot_detection` 目录，因为 `config.yaml` 中的 `path: dataset` 是相对路径。

### 批量预测

`predict.py` 默认读取：

```text
Bleeding_Spot_detection/best.pt
Bleeding_Spot_detection/Originaldata/Testing Set
```

因此训练完成后，需要把最佳权重复制或重命名为：

```text
Bleeding_Spot_detection/best.pt
```

然后执行：

```bash
python3 predict.py
```

输出位置：

```text
Bleeding_Spot_detection/results/
```

每张图会生成：

| 文件 | 含义 |
|---|---|
| `{图像名}_HE_result.tif` | 出血点二值掩码 |
| `{图像名}_visual.jpg` | 原图叠加红色出血区域和绿色检测框 |

## GUI 推理工具

GUI 位于：

```text
Software/inference_gui.py
```

该目录应包含：

```text
Software/
├── inference_gui.py
├── att_fusion_best.pth
└── best.pt
```

仓库中已通过 Git LFS 管理这两个模型权重。如果本地看不到真实权重，或模型加载时报错，请先执行：

```bash
git lfs pull
```

也可以在 GUI 界面的“模型文件路径”区域手动选择其他权重文件。

启动 GUI：

```bash
cd Vascular_Imaging_Project/Software
python3 inference_gui.py
```

GUI 支持两类任务：

| 区域 | 输入 | 输出 |
|---|---|---|
| 出血点检测 | CFP 图像文件或文件夹 | `{图像名}_HE_result.{格式}` |
| 血管分割 | CFP、OCTA 二选一，或 CFP+OCTA 同时输入 | `{图像名}_vessel_result.{格式}` |

支持输入格式：

```text
jpg, jpeg, png, tif, tiff, bmp
```

支持输出格式：

```text
tif, png, jpg, bmp
```

血管分割支持三种模式：

| 模式 | 填写方式 | 说明 |
|---|---|---|
| 双模态 | 同时填写 CFP 和 OCTA | 文件模式要求两个输入都是文件；文件夹模式要求两个输入都是文件夹，且同名文件自动配对 |
| 仅 CFP | 只填写 CFP | 代码会用 CFP 灰度图填充 OCTA 通道 |
| 仅 OCTA | 只填写 OCTA | 代码会复制 OCTA 灰度图生成 CFP 三通道 |

注意：当前 GUI 保存的是二值掩码文件，不保存原图叠加可视化图。出血点叠加可视化图由 `Bleeding_Spot_detection/predict.py` 生成。

## 软件跨平台打包方法
安装依赖：pip install pyinstaller

### macOS 打包就在Mac上执行，Windows 打包就在Windows上执行（确保模型名正确）

```bash
pyinstaller --windowed \
--name "Processing_Software" \
--add-data "att_fusion_best.pth:." \
--add-data "best.pt:." \
inference_gui.py
```

## 权重文件对应关系

| 权重文件 | 使用位置 | 生成方式 |
|---|---|---|
| `cfp_unet_best.pth` | CFP 单模态测试、多模态预训练 | 运行 CFP 模块训练 |
| `best_octa_unet.pth` | OCTA 单模态测试 | 运行 OCTA 模块训练 |
| `octa_unet_best.pth` | 多模态预训练 | 复制 `best_octa_unet.pth` 后重命名 |
| `att_fusion_best.pth` | 多模态测试、GUI 血管分割 | 运行多模态模块训练，或使用 `Software` 中已有权重 |
| `best.pt` | 出血点预测、GUI 出血点检测 | 运行 YOLOv8-Seg 训练，或使用 `Software` 中已有权重 |

如果要把自己训练的模型用于 GUI：

```text
MultiModal_Vessel_Project/saved_models/att_fusion_best.pth
→ Software/att_fusion_best.pth

Bleeding_Spot_detection/runs/segment/train/weights/best.pt
→ Software/best.pt
```

## 常见问题

### 模型文件加载失败

先确认是否拉取了 Git LFS 文件：

```bash
git lfs pull
```

再确认文件名是否与代码一致：

```text
Software/att_fusion_best.pth
Software/best.pt
Bleeding_Spot_detection/best.pt
```

### OCTA 模块第一次运行就报找不到权重

`OCTA_Vessel_Project/main.py` 默认是测试模式：

```python
RUN_TRAIN = False
RUN_TEST = True
```

首次运行前需要先把 `RUN_TRAIN` 改为 `True`，或提前放入 `saved_model/best_octa_unet.pth`。

### 多模态训练提示找不到预训练权重

默认配置会加载：

```text
data/cfp_pretrained/cfp_unet_best.pth
data/octa_pretrained/octa_unet_best.pth
```

如果没有这两个文件，要么先训练 CFP 和 OCTA 单模态模型，要么在 `code/main.py` 中关闭预训练：

```python
LOAD_PRETRAINED = False
FREEZE_ENCODER = False
```

### GUI 双文件夹模式没有结果

双文件夹模式按不带扩展名的文件名取交集配对。例如：

```text
cfp/001.png
octa/001.png
```

可以配对；但下面这种不能配对：

```text
cfp/CFP_001.png
octa/OCTA_001.png
```

### YOLO 训练找不到数据

`config.yaml` 中写的是：

```yaml
path: dataset
train: images/train
val: images/val
```

因此直接使用 `yolo` 命令时，当前目录必须是 `Bleeding_Spot_detection`。

### 路径中有空格或中文

代码内部使用相对路径和 `os.path.abspath(__file__)` 定位，一般可以处理中文路径。终端中手动 `cd` 时，如果路径包含空格，需要加引号：

```bash
cd "/Users/yourname/Desktop/Vascular Imaging Project/Software"
```
