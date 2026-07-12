import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import imageio
import re

class DriveEyeDataset(Dataset):
    def __init__(self, state, root=None, transform=None, target_transform=None):
        """
        state: 'train' / 'val' / 'test'
        root: 数据集DRIVE根目录，由main自动传入，解决路径错乱
        transform: 输入图像预处理
        target_transform: 标签预处理
        """
        self.state = state
        self.aug = True
        # 优先使用外部传入的项目根，不传才使用旧相对路径兜底
        if root is None:
            self.root = './datasets/DRIVE'
        else:
            self.root = root
        self.img_paths = []
        self.mask_paths = []
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        # ========== 训练集配对 ==========
        train_img_dir = os.path.join(self.root, 'training', 'images')
        train_mask_dir = os.path.join(self.root, 'training', '1st_manual')
        train_img_files = os.listdir(train_img_dir)
        train_img_dict = {}
        for f in train_img_files:
            num = re.findall(r'\d+', f)[0]
            train_img_dict[num] = os.path.join(train_img_dir, f)
        train_mask_files = os.listdir(train_mask_dir)
        train_mask_dict = {}
        for f in train_mask_files:
            num = re.findall(r'\d+', f)[0]                # 修复：提取列表第一个元素，避免不可哈希错误
            train_mask_dict[num] = os.path.join(train_mask_dir, f)
        train_common_nums = sorted(train_img_dict.keys() & train_mask_dict.keys(), key=int)
        train_img = [train_img_dict[n] for n in train_common_nums]
        train_mask = [train_mask_dict[n] for n in train_common_nums]

        # ========== 测试集配对 ==========
        test_img_dir = os.path.join(self.root, 'test', 'images')
        test_mask_dir = os.path.join(self.root, 'test', '1st_manual')
        test_img_files = os.listdir(test_img_dir)
        test_img_dict = {}
        for f in test_img_files:
            num = re.findall(r'\d+', f)[0]
            test_img_dict[num] = os.path.join(test_img_dir, f)
        test_mask_files = os.listdir(test_mask_dir)
        test_mask_dict = {}
        for f in test_mask_files:
            num = re.findall(r'\d+', f)[0]                # 修复：提取列表第一个元素
            test_mask_dict[num] = os.path.join(test_mask_dir, f)
        test_common_nums = sorted(test_img_dict.keys() & test_mask_dict.keys(), key=int)
        test_img = [test_img_dict[n] for n in test_common_nums]
        test_mask = [test_mask_dict[n] for n in test_common_nums]

        if self.state == 'train':
            return train_img, train_mask
        else:
            return test_img, test_mask

    def __getitem__(self, index):
        img_path = self.pics[index]
        mask_path = self.masks[index]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 兼容gif标注
        if mask is None:
            mask_frames = imageio.mimread(mask_path)
            mask = np.array(mask_frames[0])
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        img = cv2.resize(img, (256, 256))
        # 修复：传入标准二元尺寸(256,256)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, img_path, mask_path

    def __len__(self):
        return len(self.pics)