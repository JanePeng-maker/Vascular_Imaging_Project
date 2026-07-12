import os
import cv2
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OCTA500Dataset(Dataset):
    def __init__(self, data_root, split='train', num_samples=100, img_size=(256, 256)):
        super().__init__()
        self.img_size = img_size

        if split == 'train':
            img_dir = os.path.join(data_root, 'train_images')
            mask_dir = os.path.join(data_root, 'train_masks')
        else:
            img_dir = os.path.join(data_root, 'test_images')
            mask_dir = os.path.join(data_root, 'test_masks')

        self.img_paths = sorted(glob(os.path.join(img_dir, '*.png')))[:num_samples]
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))[:num_samples]

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        filename = os.path.basename(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = (mask_gray > 127).astype(np.float32)

        img_tensor = self.img_transform(img)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return img_tensor, mask_tensor, filename