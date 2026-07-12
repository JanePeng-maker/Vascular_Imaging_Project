import argparse
import logging
import os
import sys
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
from torchvision.transforms import transforms
from scipy.spatial.distance import directed_hausdorff
from nets.UNet import Unet
from nets.attention_unet import AttU_Net
from nets.r2unet import R2U_Net
from nets.segnet import SegNet
from nets.unetpp import NestedUNet
from nets.fcn import get_fcn8s
from dataset import DriveEyeDataset
# 修复：函数名匹配metrics_plot
from plot import loss_plot, metrics_plot

# 固定随机种子，保证验证集划分可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 自动锁定当前脚本所在UNets-master目录为项目根
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_PATH)
sys.path.append(PROJECT_ROOT)

# 全局路径全部基于脚本根拼接，无硬编码
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
PREDICT_DIR = os.path.join(PROJECT_ROOT, "saved_predict")

# 自动创建输出文件夹
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PREDICT_DIR, exist_ok=True)

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0, type=int)
    parse.add_argument("--action", type=str, default="train&test",
                       choices=['train', 'test', 'train&test'],
                       help="执行动作：train（仅训练）, test（仅测试）, train&test（训练+测试）")
    parse.add_argument("--epoch", type=int, default=10, help="训练轮次")
    parse.add_argument('--arch', '-a', default='UNet',
                       help='UNet/resnet34_unet/unet++/Attention_UNet/segnet/r2unet/fcn8s')
    parse.add_argument("--batch_size", type=int, default=1, help="batch size（CPU下建议1）")
    parse.add_argument('--dataset', default='driveEye', help="固定为driveEye")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    parse.add_argument("--model_path", type=str, default=None,
                       help="指定要加载的模型权重文件路径（用于测试），例如 ./saved_model/cfp_unet_best.pth")
    parse.add_argument("--num_test", type=int, default=-1,
                       help="测试时只处理前 N 张图片，-1 表示全部")
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(PROJECT_ROOT, args.log_dir, args.arch, str(args.batch_size), args.dataset, str(args.epoch))
    filename = os.path.join(dirname, 'log.log')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logging

def getModel(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)
    elif args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args, 3, 1).to(device)
    elif args.arch == 'Attention_UNet':
        model = AttU_Net(3, 1).to(device)
    elif args.arch == 'segnet':
        model = SegNet(3, 1).to(device)
    elif args.arch == 'r2unet':
        model = R2U_Net(3, 1).to(device)
    elif args.arch == 'fcn8s':
        model = get_fcn8s(1).to(device)
    else:
        raise ValueError(f"Unsupported arch: {args.arch}")
    return model

def getDataset(args, num_test_samples=-1):
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        # 修复：3通道RGB匹配三组均值方差
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y_transforms = transforms.ToTensor()
    # 修复：传入数据集根路径，解决跨目录找不到数据
    train_dataset = DriveEyeDataset('train', root=DATASET_ROOT, transform=x_transforms, target_transform=y_transforms)
    test_dataset = DriveEyeDataset('test', root=DATASET_ROOT, transform=x_transforms)

    # 从训练集中划分出验证集（20%），避免验证集与测试集重合导致数据泄露
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    if num_test_samples > 0:
        test_dataset = Subset(test_dataset, list(range(min(num_test_samples, len(test_dataset)))))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader

# ---------- 指标计算函数 ----------
def get_iou_from_arrays(mask, pred):
    mask = (mask > 0.5).astype(np.uint8)
    pred = (pred > 0.5).astype(np.uint8)
    inter = np.logical_and(mask, pred).sum()
    union = np.logical_or(mask, pred).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union

def get_dice_from_arrays(mask, pred):
    mask = (mask > 0.5).astype(np.uint8)
    pred = (pred > 0.5).astype(np.uint8)
    inter = np.logical_and(mask, pred).sum()
    if mask.sum() + pred.sum() == 0:
        return 1.0
    return 2.0 * inter / (mask.sum() + pred.sum())

def get_hd_from_arrays(mask, pred):
    mask_pts = np.argwhere(mask > 0.5)
    pred_pts = np.argwhere(pred > 0.5)
    if len(mask_pts) == 0 or len(pred_pts) == 0:
        return 0.0
    hd1 = directed_hausdorff(mask_pts, pred_pts)[0]
    hd2 = directed_hausdorff(pred_pts, mask_pts)[0]
    return max(hd1, hd2)

def val(model, best_iou, val_loader, args, device, logging):
    model.eval()
    with torch.no_grad():
        miou_total, hd_total, dice_total = 0, 0, 0
        num = len(val_loader)
        for img, mask, _, _ in val_loader:
            img = img.to(device)
            output = model(img)
            if args.deepsupervision:
                pred = torch.squeeze(output[-1]).cpu().numpy()
            else:
                pred = torch.squeeze(output).cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)
            mask_np = mask.squeeze().numpy()
            miou_total += get_iou_from_arrays(mask_np, pred_binary)
            hd_total += get_hd_from_arrays(mask_np, pred_binary)
            dice_total += get_dice_from_arrays(mask_np, pred_binary)
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total / num
        print(f'Val: Miou={aver_iou:.4f}, HD={aver_hd:.4f}, Dice={aver_dice:.4f}')
        logging.info(f'Val: Miou={aver_iou:.4f}, HD={aver_hd:.4f}, Dice={aver_dice:.4f}')
        if aver_iou > best_iou:
            best_iou = aver_iou
            torch.save(model.state_dict(), os.path.join(SAVE_MODEL_DIR, "cfp_unet_best.pth"))

            print('==> Save best model')
            logging.info('==> Save best model')
        return best_iou, aver_iou, aver_dice, aver_hd

def train(model, criterion, optimizer, train_loader, val_loader, args, device, logging):
    best_iou = 0
    loss_list, iou_list, dice_list, hd_list = [], [], [], []
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0
        for step, (img, mask, _, _) in enumerate(train_loader, 1):
            img = img.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(img)
                loss = sum(criterion(out, mask) for out in outputs) / len(outputs)
            else:
                output = model(img)
                loss = criterion(output, mask)
            if args.threshold is None or loss.item() > args.threshold:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epoch}, Avg Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} loss={avg_loss:.4f}")
        best_iou, aver_iou, aver_dice, aver_hd = val(model, best_iou, val_loader, args, device, logging)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice', iou_list, dice_list)
    metrics_plot(args, 'hd', hd_list)
    return model

def test(test_loader, args, device, logging, model_path=None, save_predict=True):
    logging.info('Final test...')
    model = getModel(args)
    default_path = os.path.join(SAVE_MODEL_DIR, "cfp_unet_best.pth")
    load_path = model_path if (model_path and os.path.exists(model_path)) else default_path
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    save_dir = os.path.join(PREDICT_DIR, args.arch, str(args.batch_size), str(args.dataset), str(args.epoch))
    if save_predict:
        os.makedirs(save_dir, exist_ok=True)
    miou_total, hd_total, dice_total = 0, 0, 0
    num = len(test_loader)
    with torch.no_grad():
        for i, (img, mask, img_path, mask_path) in enumerate(test_loader):
            img = img.to(device)
            pred = model(img)
            if args.deepsupervision:
                pred_np = torch.squeeze(pred[-1]).cpu().numpy()
            else:
                pred_np = torch.squeeze(pred).cpu().numpy()
            pred_binary = (pred_np > 0.5).astype(np.float32)
            mask_np = mask.squeeze().numpy()
            iou = get_iou_from_arrays(mask_np, pred_binary)
            dice = get_dice_from_arrays(mask_np, pred_binary)
            hd = get_hd_from_arrays(mask_np, pred_binary)
            miou_total += iou
            dice_total += dice
            hd_total += hd
            print(f"Test {i + 1}: IoU={iou:.4f}, Dice={dice:.4f}, HD={hd:.4f}")
            if save_predict:
                img_orig = cv2.imread(img_path[0])
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_orig)
                axes[0].set_title('Input')
                axes[0].axis('off')
                axes[1].imshow(pred_binary, cmap='gray')
                axes[1].set_title('Pred Mask')
                axes[1].axis('off')
                axes[2].imshow(mask_np, cmap='gray')
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
                plt.tight_layout()
                save_name = os.path.splitext(os.path.basename(mask_path[0]))[0] + '.png'
                plt.savefig(os.path.join(save_dir, save_name), dpi=100)
                plt.close()
    avg_iou = miou_total / num
    avg_dice = dice_total / num
    avg_hd = hd_total / num          # 修复：HD除以图片总数，得到平均值
    print(f"\nTest Results: Avg IoU={avg_iou:.4f}, Avg Dice={avg_dice:.4f}, Avg HD={avg_hd:.4f}")
    logging.info(f"Test Results: Avg IoU={avg_iou:.4f}, Avg Dice={avg_dice:.4f}, Avg HD={avg_hd:.4f}")

if __name__ == "__main__":
    args = getArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging = getLog(args)
    print('=' * 50)
    print(f"Model: {args.arch}, Epochs: {args.epoch}, Batch Size: {args.batch_size}, Dataset: {args.dataset}")
    logging.info(f"Model: {args.arch}, Epochs: {args.epoch}, Batch Size: {args.batch_size}, Dataset: {args.dataset}")
    train_loader, val_loader, test_loader = getDataset(args, num_test_samples=args.num_test)
    if 'train' in args.action:
        model = getModel(args)
        pos_weight = torch.tensor([10.0]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        train(model, criterion, optimizer, train_loader, val_loader, args, device, logging)
    if 'test' in args.action:
        test(test_loader, args, device, logging, model_path=args.model_path, save_predict=True)