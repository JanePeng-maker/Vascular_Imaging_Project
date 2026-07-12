import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==================== 自动定位项目根目录 ====================
# 获取当前脚本所在目录（即 main.py 所在的文件夹）
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
# 将项目根目录添加到 sys.path，确保可以导入同目录下的 model 和 dataset
sys.path.append(PROJECT_ROOT)
# ============================================================

# 导入自定义模块（现在可以正确找到）
from model import UNet
from dataset import OCTA500Dataset

# ==================================================
# ==========  全局可配置参数（直接改这里即可）  ==========
# ==================================================
# 1. 数据集根目录（基于项目根目录拼接，无需修改，直接使用子文件夹）
DATA_ROOT = PROJECT_ROOT   # 因为 train_images 等文件夹与 main.py 同级
# 2. 训练样本数量（默认100）
N_TRAIN_SAMPLES = 100
# 3. 测试样本数量（默认5）
N_TEST_SAMPLES = 5
# 4. 图像统一尺寸（建议保持2的倍数，CPU推荐256或512）
IMG_SIZE = (256, 256)
# 5. 训练轮次（CPU建议30轮）
EPOCHS = 30
# 6. 批次大小（CPU强烈建议保持为1）
BATCH_SIZE = 1
# 7. 学习率
LEARNING_RATE = 1e-4
# 8. 正样本权重（血管占比小，加权缓解类别不平衡）
POS_WEIGHT = 10.0
# 9. 运行设备：自动检测GPU，没有就用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 10. 运行开关
RUN_TRAIN = False   # 是否训练
RUN_TEST = True     # 是否测试
# ==================================================

# 定义保存路径（均基于项目根目录）
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
TEST_RESULT_DIR = os.path.join(PROJECT_ROOT, "test_results")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "best_octa_unet.pth")
LOSS_CURVE_PATH = os.path.join(SAVED_MODEL_DIR, "loss_curve.png")


def train():
    """训练函数"""
    print("=" * 50)
    print(f"开始训练，运行设备：{DEVICE}")
    print(f"训练样本数：{N_TRAIN_SAMPLES}，训练轮次：{EPOCHS}")
    print("=" * 50)

    # 1. 加载训练数据集
    train_dataset = OCTA500Dataset(
        data_root=DATA_ROOT,
        split='train',
        num_samples=N_TRAIN_SAMPLES,
        img_size=IMG_SIZE
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 初始化模型、损失函数、优化器
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 创建模型保存文件夹
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    # 4. 训练循环
    loss_list = []
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        step = 0

        for imgs, masks, _ in train_loader:
            step += 1
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / step
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  平均损失: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("  → 保存当前最优模型")

    # 5. 绘制并保存损失曲线
    plt.figure()
    plt.plot(range(1, EPOCHS+1), loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH, dpi=150)
    plt.close()

    print("\n训练完成！")
    print(f"最优模型保存路径：{MODEL_PATH}")
    print(f"损失曲线保存路径：{LOSS_CURVE_PATH}")


def test():
    """测试函数：生成分割结果并保存对比图"""
    print("\n" + "=" * 50)
    print(f"开始测试，测试样本数：{N_TEST_SAMPLES}")
    print("=" * 50)

    # 1. 创建结果保存文件夹
    os.makedirs(TEST_RESULT_DIR, exist_ok=True)

    # 2. 加载测试数据集
    test_dataset = OCTA500Dataset(
        data_root=DATA_ROOT,
        split='test',
        num_samples=N_TEST_SAMPLES,
        img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 加载训练好的最优模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型权重文件：{MODEL_PATH}，请先运行训练（RUN_TRAIN=True）")

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 4. 逐张推理并保存对比图
    with torch.no_grad():
        for i, (img, mask, filename) in enumerate(test_loader):
            img = img.to(DEVICE)
            pred_logits = model(img)
            pred_prob = torch.sigmoid(pred_logits)

            img_np = img.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = pred_prob.squeeze().cpu().numpy()
            pred_binary = (pred_np > 0.5).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np, cmap='gray')
            axes[0].set_title('OCTA Image')
            axes[0].axis('off')

            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            save_path = os.path.join(TEST_RESULT_DIR, f"result_{filename[0]}")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"已保存第 {i+1} 张结果：{filename[0]}")

    print(f"\n测试完成！所有结果已保存到：{TEST_RESULT_DIR}")


if __name__ == "__main__":
    if RUN_TRAIN:
        train()
    if RUN_TEST:
        test()