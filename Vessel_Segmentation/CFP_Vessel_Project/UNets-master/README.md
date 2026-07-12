markdown
# CFP_Vessel_Project —— 眼底血管分割（基于 DRIVE 数据集）

本项目基于 PyTorch 实现多种 UNet 变体（UNet、Attention-UNet、UNet++、SegNet 等），用于眼底视网膜血管分割。您可以使用预训练模型或从头训练，在 DRIVE 数据集上进行测试。

## 环境要求
- Python 3.8+
- PyTorch 1.10+
- OpenCV, NumPy, Matplotlib, SciPy, ImageIO

## 安装依赖
在终端中执行：
pip install torch torchvision opencv-python numpy matplotlib scipy imageio

## 使用说明

1. 从头训练并测试

训练 UNet（默认 10 轮）并自动测试：
bash
python main.py --action train&test

2. 仅测试（使用已有模型）

若已有模型文件（如 cfp_unet_best.pth），跳过训练：
bash
python main.py --action test --model_path ./saved_model/cfp_unet_best.pth
默认测试全部测试集图片。
若只需测试前 10 张，加 --num_test 10。

3. 更换网络结构

支持 UNet（默认）、Attention_UNet、unet++、segnet、r2unet、fcn8s。
示例（训练 Attention-UNet 20 轮）：
bash
python main.py --action train&test --arch Attention_UNet --epoch 20

4. 调整训练参数

--epoch : 训练轮数（默认 10）
--batch_size : 批次大小（CPU 建议 1，GPU 可适当增大）
--lr : 学习率（目前固定为 1e-4，可在 main.py 中修改）
输出结果

训练日志：保存在 result/log/ 下。
损失曲线：保存在 result/plot/ 下。
最佳模型：保存在 saved_model/，命名格式为 {arch}_{batch_size}_{dataset}_{epoch}.pth。
测试对比图：保存在 saved_predict/{arch}/{batch_size}/{epoch}/{dataset}/，每张图片为三栏对比（输入图、预测 Mask、真实 Mask），格式为 PNG。

## 使用时常见问题

Q: 测试时提示找不到模型文件？
A: 请确保 --model_path 指定的路径正确，或者训练参数（--arch、--epoch 等）与保存的模型文件名一致。

Q: 训练时 loss 不下降？
A: 可尝试调整正样本权重（代码中 pos_weight 参数，默认 10.0，可增至 15~20），或降低学习率。

Q: 如何只测试前 N 张图？
A: 使用 --num_test N。

Q: 我的图像不是 .tif 或 .gif 怎么办？
A: 请确保您的图片格式能被 OpenCV 读取，或在 dataset.py 中修改 cv2.imread 部分。

## 数据集

the Eye Vessels(DRIVE dataset)
link：https://pan.baidu.com/s/1UkMLmdbM61N8ecgnKlAsPg 
keyword：f1ek
