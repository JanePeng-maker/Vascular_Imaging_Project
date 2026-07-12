import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from model import EarlyFusionUNet, AttentionFusionUNet
from dataset import MultiModalVesselDataset

# 1. 数据集路径（相对于code目录，无需修改）
DATA_ROOT = "../data/pseudo_pair"

# 2. 模型类型：att_fusion（注意力融合，推荐） / early_fusion（早期融合基线）
MODEL_TYPE = "att_fusion"

# 3. 图像尺寸（与生成数据集保持一致，请勿修改）
IMG_SIZE = (256, 256)

# 4. 训练超参数
EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
POS_WEIGHT = 10.0  # 血管正样本权重，缓解类别不平衡

# 5. 核心方案：加载单模态预训练权重 + 冻结编码器主干
LOAD_PRETRAINED = True    # 是否加载单模态预训练权重
FREEZE_ENCODER = True     # 是否冻结单模态编码器（True=完全不重训单模型）
CFP_WEIGHT_PATH = "../data/cfp_pretrained/cfp_unet_best.pth"
OCTA_WEIGHT_PATH = "../data/octa_pretrained/octa_unet_best.pth"

# 6. 测试参数
NUM_TEST_SAMPLES = 5  # 测试前N张，-1表示全部测试

# 7. 运行开关
RUN_TRAIN = False
RUN_TEST = True

# 8. 自动选择运行设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================


def compute_metrics(mask_np, pred_np):
    """计算分割指标：IoU、Dice"""
    mask = (mask_np > 0.5).astype(np.uint8)
    pred = (pred_np > 0.5).astype(np.uint8)
    inter = np.logical_and(mask, pred).sum()
    union = np.logical_or(mask, pred).sum()

    iou = inter / union if union > 0 else 1.0
    dice = 2 * inter / (mask.sum() + pred.sum()) if (mask.sum() + pred.sum()) > 0 else 1.0
    return iou, dice


def load_pretrained_weights(model):
    """加载CFP、OCTA单模态预训练权重，初始化双流编码器"""
    if not LOAD_PRETRAINED:
        return model

    print("\n" + "=" * 50)
    print("正在加载单模态预训练权重...")
    model_dict = model.state_dict()

    # ========== 加载CFP预训练权重 ==========
    if os.path.exists(CFP_WEIGHT_PATH):
        cfp_pretrained = torch.load(CFP_WEIGHT_PATH, map_location=DEVICE)
        # 层名映射：单模态UNet层名 → 多模态CFP分支层名
        cfp_layer_map = {
            'inc.double_conv': 'cfp_inc.double_conv',
            'down1.maxpool_conv': 'cfp_down1.maxpool_conv',
            'down2.maxpool_conv': 'cfp_down2.maxpool_conv',
            'down3.maxpool_conv': 'cfp_down3.maxpool_conv',
        }
        loaded_count = 0
        for old_key, value in cfp_pretrained.items():
            for old_prefix, new_prefix in cfp_layer_map.items():
                if old_prefix in old_key:
                    new_key = old_key.replace(old_prefix, new_prefix)
                    if new_key in model_dict and value.shape == model_dict[new_key].shape:
                        model_dict[new_key] = value
                        loaded_count += 1
                    break
        print(f"  ✅ CFP编码器加载成功：{loaded_count} 层参数")
    else:
        print(f"  ⚠️  未找到CFP预训练权重：{CFP_WEIGHT_PATH}")

    # ========== 加载OCTA预训练权重 ==========
    if os.path.exists(OCTA_WEIGHT_PATH):
        octa_pretrained = torch.load(OCTA_WEIGHT_PATH, map_location=DEVICE)
        # 层名映射：单模态UNet层名 → 多模态OCTA分支层名
        octa_layer_map = {
            'inc.double_conv': 'octa_inc.double_conv',
            'down1.maxpool_conv': 'octa_down1.maxpool_conv',
            'down2.maxpool_conv': 'octa_down2.maxpool_conv',
            'down3.maxpool_conv': 'octa_down3.maxpool_conv',
        }
        loaded_count = 0
        for old_key, value in octa_pretrained.items():
            for old_prefix, new_prefix in octa_layer_map.items():
                if old_prefix in old_key:
                    new_key = old_key.replace(old_prefix, new_prefix)
                    if new_key in model_dict and value.shape == model_dict[new_key].shape:
                        model_dict[new_key] = value
                        loaded_count += 1
                    break
        print(f"  ✅ OCTA编码器加载成功：{loaded_count} 层参数")
    else:
        print(f"  ⚠️  未找到OCTA预训练权重：{OCTA_WEIGHT_PATH}")

    model.load_state_dict(model_dict)
    print("=" * 50 + "\n")
    return model


def freeze_encoder_layers(model):
    """冻结CFP、OCTA两个编码器主干，参数完全不参与训练"""
    if not FREEZE_ENCODER:
        return model

    frozen_count = 0
    for name, param in model.named_parameters():
        # 冻结所有以 cfp_ 和 octa_ 开头的编码器层
        if name.startswith('cfp_') or name.startswith('octa_'):
            param.requires_grad = False
            frozen_count += 1

    print(f"🔒 已冻结单模态编码器：共 {frozen_count} 层参数固定不训练")
    # 统计参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数量：{total_params / 1e6:.2f}M | 可训练参数量：{trainable_params / 1e6:.2f}M\n")
    return model


def train():
    """训练函数：含验证、最优模型保存、训练曲线"""
    print("=" * 60)
    print(f"开始训练 | 模型：{MODEL_TYPE} | 轮次：{EPOCHS} | 设备：{DEVICE}")
    if FREEZE_ENCODER and MODEL_TYPE == "att_fusion":
        print("模式：冻结单模态编码器，仅训练融合模块 + 解码器")
    print("=" * 60)

    # 1. 加载数据集
    train_dataset = MultiModalVesselDataset(DATA_ROOT, 'train', IMG_SIZE)
    val_dataset = MultiModalVesselDataset(DATA_ROOT, 'test', IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. 初始化模型 + 加载权重 + 冻结编码器
    if MODEL_TYPE == 'early_fusion':
        model = EarlyFusionUNet(n_classes=1).to(DEVICE)
        print("⚠️  早期融合模型不支持双流预训练加载与冻结")
    else:
        model = AttentionFusionUNet(n_classes=1).to(DEVICE)
        model = load_pretrained_weights(model)
        model = freeze_encoder_layers(model)

    # 3. 损失函数与优化器（仅优化可训练参数）
    pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # 4. 创建保存目录
    os.makedirs("../saved_models", exist_ok=True)
    save_name = f"{MODEL_TYPE}_best.pth"

    # 5. 训练循环
    train_loss_list = []
    val_dice_list = []
    best_dice = 0.0

    for epoch in range(EPOCHS):
        # ---- 训练阶段 ----
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

        avg_train_loss = epoch_loss / step
        train_loss_list.append(avg_train_loss)

        # ---- 验证阶段 ----
        model.eval()
        total_iou = 0
        total_dice = 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(imgs)
                pred_prob = torch.sigmoid(outputs)

                pred_np = pred_prob.squeeze().cpu().numpy()
                mask_np = masks.squeeze().cpu().numpy()
                iou, dice = compute_metrics(mask_np, pred_np)
                total_iou += iou
                total_dice += dice

        avg_val_iou = total_iou / len(val_loader)
        avg_val_dice = total_dice / len(val_loader)
        val_dice_list.append(avg_val_dice)

        # 打印本轮结果
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"  训练损失: {avg_train_loss:.4f} | 验证IoU: {avg_val_iou:.4f} | 验证Dice: {avg_val_dice:.4f}")

        # 保存Dice最高的最优模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join("../saved_models", save_name))
            print("  → 更新最优模型")

    # 6. 绘制并保存训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), val_dice_list, label='Val Dice', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.title('Validation Dice')

    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/{MODEL_TYPE}_training_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n训练完成！最优模型已保存：../saved_models/{save_name}")
    print(f"训练曲线已保存：../results/{MODEL_TYPE}_training_curve.png")
    print(f"最优验证Dice: {best_dice:.4f}")


def test():
    """测试函数：生成分割对比图 + 计算平均指标"""
    print("\n" + "=" * 60)
    print(f"开始测试 | 模型：{MODEL_TYPE} | 测试样本数：{NUM_TEST_SAMPLES}")
    print("=" * 60)

    # 1. 加载测试数据集
    test_dataset = MultiModalVesselDataset(DATA_ROOT, 'test', IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. 加载最优模型
    if MODEL_TYPE == 'early_fusion':
        model = EarlyFusionUNet(n_classes=1).to(DEVICE)
    else:
        model = AttentionFusionUNet(n_classes=1).to(DEVICE)

    model_path = os.path.join("../saved_models", f"{MODEL_TYPE}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}，请先运行训练")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. 创建结果保存目录
    save_dir = f"../results/{MODEL_TYPE}"
    os.makedirs(save_dir, exist_ok=True)

    # 4. 推理并保存结果
    total_iou = 0
    total_dice = 0
    test_count = min(NUM_TEST_SAMPLES, len(test_loader)) if NUM_TEST_SAMPLES > 0 else len(test_loader)

    with torch.no_grad():
        for i, (img, mask, filename) in enumerate(test_loader):
            if i >= test_count:
                break

            img = img.to(DEVICE)
            pred_logits = model(img)
            pred_prob = torch.sigmoid(pred_logits)

            # 转numpy格式
            img_np = img.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = pred_prob.squeeze().cpu().numpy()
            pred_binary = (pred_np > 0.5).astype(np.uint8)

            # 计算指标
            iou, dice = compute_metrics(mask_np, pred_np)
            total_iou += iou
            total_dice += dice

            # 绘制三联对比图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # 取前3通道显示CFP原图
            cfp_show = img_np[:3].transpose(1, 2, 0) * 0.5 + 0.5
            axes[0].imshow(cfp_show)
            axes[0].set_title('Input CFP')
            axes[0].axis('off')

            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title(f'Prediction (Dice={dice:.3f})')
            axes[2].axis('off')

            save_path = os.path.join(save_dir, f"result_{filename[0]}")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"[{i + 1}/{test_count}] {filename[0]} | IoU={iou:.4f} | Dice={dice:.4f}")

    avg_iou = total_iou / test_count
    avg_dice = total_dice / test_count
    print(f"\n测试完成！平均指标：IoU={avg_iou:.4f} | Dice={avg_dice:.4f}")
    print(f"结果已保存到：{save_dir}")


if __name__ == "__main__":
    if RUN_TRAIN:
        train()
    if RUN_TEST:
        test()