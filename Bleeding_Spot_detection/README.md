# 眼底出血点检测（YOLOv8-Seg）

基于 YOLOv8 实例分割模型，使用 IDRiD 数据集训练眼底出血点（Hemorrhage, HE）检测模型。
支持 jpg / png / tif 多格式输入，输出 tif 格式二值分割掩码。

## 环境要求
- Python 3.8 ~ 3.12
- 推荐 NVIDIA GPU（CUDA），CPU 也可运行但较慢

## 运行步骤

在项目根目录下打开终端，执行：

### 1. 进入项目目录
bash
cd ~/Desktop/眼底出血点检测(替换为使用者实际路径，需要在所有代码中的全局变量处同步修改)

### 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

### 3. 安装依赖
pip install ultralytics==8.2.0 opencv-python==4.9.0.80 numpy==1.26.4 tifffile==2024.2.12 PyYAML==6.0.1