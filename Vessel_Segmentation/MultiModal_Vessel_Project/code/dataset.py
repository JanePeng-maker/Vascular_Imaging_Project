import os
import cv2
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiModalVesselDataset(Dataset):
    """
    多模态血管分割数据集
    输入：CFP彩色图 + OCTA灰度图（像素级对齐）
    输出：4通道拼接张量 + 血管二值mask + 文件名
    """
    def __init__(self, data_root, split='train', img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size
        split_dir = os.path.join(data_root, split)

        # 三个文件夹文件名完全一致，排序后严格对应
        self.cfp_paths = sorted(
            glob(os.path.join(split_dir, 'cfp', '*.png')) +
            glob(os.path.join(split_dir, 'cfp', '*.tif')) +
            glob(os.path.join(split_dir, 'cfp', '*.gif'))
        )
        self.octa_paths = sorted(
            glob(os.path.join(split_dir, 'octa', '*.png')) +
            glob(os.path.join(split_dir, 'octa', '*.tif'))
        )
        self.mask_paths = sorted(
            glob(os.path.join(split_dir, 'mask', '*.png')) +
            glob(os.path.join(split_dir, 'mask', '*.tif')) +
            glob(os.path.join(split_dir, 'mask', '*.gif'))
        )

        # 预处理：与单模态项目完全对齐
        self.cfp_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.octa_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.cfp_paths)

    def __getitem__(self, idx):
        filename = os.path.basename(self.cfp_paths[idx])

        # 1. 读取并预处理CFP彩色图
        cfp = cv2.imread(self.cfp_paths[idx])
        cfp = cv2.cvtColor(cfp, cv2.COLOR_BGR2RGB)
        cfp = cv2.resize(cfp, self.img_size)
        cfp = cfp.astype(np.float32) / 255.0
        cfp_tensor = self.cfp_transform(cfp)

        # 2. 读取并预处理OCTA灰度图
        octa = cv2.imread(self.octa_paths[idx], cv2.IMREAD_GRAYSCALE)
        octa = cv2.resize(octa, self.img_size)
        octa = octa.astype(np.float32) / 255.0
        octa_tensor = self.octa_transform(octa)

        # 3. 读取并预处理血管mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask_binary = np.zeros_like(mask, dtype=np.float32)
        mask_binary[mask > 127] = 1.0
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        # 4. 通道维度拼接：3通道CFP + 1通道OCTA = 4通道输入
        input_tensor = torch.cat([cfp_tensor, octa_tensor], dim=0)

        return input_tensor, mask_tensor, filename