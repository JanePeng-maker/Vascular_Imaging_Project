import os
import cv2
import numpy as np
from skimage.exposure import match_histograms
from glob import glob
import imageio
import re

# ==================================================
# 自动获取项目根路径（code文件夹的上一级），无需手动修改
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ==================================================

# 原始数据路径（与目录结构严格对应）
CFP_ROOT = os.path.join(ROOT_DIR, 'data', 'cfp_original')
OCTA_REF_DIR = os.path.join(ROOT_DIR, 'data', 'octa_original', 'train_images')

# 输出伪配对数据集路径
OUTPUT_ROOT = os.path.join(ROOT_DIR, 'data', 'pseudo_pair')

# 统一图像尺寸，与CFP单模态项目保持完全一致
IMG_SIZE = (256, 256)
# 取前10张OCTA图计算参考灰度分布，比单张更稳定
REF_IMG_COUNT = 10


def get_octa_reference():
    """读取多张OCTA图像，计算平均图作为参考风格"""
    octa_paths = sorted(glob(os.path.join(OCTA_REF_DIR, '*.png')))[:REF_IMG_COUNT]
    if len(octa_paths) == 0:
        raise FileNotFoundError(f"未在 {OCTA_REF_DIR} 找到OCTA参考图像，请检查路径")

    imgs = []
    for p in octa_paths:
        imgs.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32))
    # 计算平均图再转回uint8
    avg_img = np.mean(imgs, axis=0).astype(np.uint8)
    return avg_img


def get_paired_paths(img_dir, mask_dir):
    """按编号严格配对图像与标注，彻底避免文件名排序错位"""
    # 提取图像的编号
    img_files = os.listdir(img_dir)
    img_dict = {}
    for f in img_files:
        num_list = re.findall(r'\d+', f)
        if len(num_list) == 0:
            continue
        num = num_list[0]
        img_dict[num] = os.path.join(img_dir, f)

    # 提取标注的编号
    mask_files = os.listdir(mask_dir)
    mask_dict = {}
    for f in mask_files:
        num_list = re.findall(r'\d+', f)
        if len(num_list) == 0:
            continue
        num = num_list[0]
        mask_dict[num] = os.path.join(mask_dir, f)

    # 取交集并按数字大小排序，保证一一对应
    common_nums = sorted(img_dict.keys() & mask_dict.keys(), key=int)
    img_paths = [img_dict[n] for n in common_nums]
    mask_paths = [mask_dict[n] for n in common_nums]
    return img_paths, mask_paths


def process_split(cfp_split_name, output_split_name, ref_hist):
    """处理单个数据折（training / test）"""
    print(f"\n正在处理 {cfp_split_name} 集...")

    # 原始CFP数据路径
    src_cfp_dir = os.path.join(CFP_ROOT, cfp_split_name, 'images')
    src_mask_dir = os.path.join(CFP_ROOT, cfp_split_name, '1st_manual')

    # 按编号严格配对
    cfp_paths, mask_paths = get_paired_paths(src_cfp_dir, src_mask_dir)
    if len(cfp_paths) != len(mask_paths) or len(cfp_paths) == 0:
        raise ValueError(f"{cfp_split_name}集：图像与标注配对失败，数量不匹配")

    # 输出路径
    out_cfp_dir = os.path.join(OUTPUT_ROOT, output_split_name, 'cfp')
    out_octa_dir = os.path.join(OUTPUT_ROOT, output_split_name, 'octa')
    out_mask_dir = os.path.join(OUTPUT_ROOT, output_split_name, 'mask')
    for d in [out_cfp_dir, out_octa_dir, out_mask_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"共找到 {len(cfp_paths)} 组样本")

    for i, (cfp_path, mask_path) in enumerate(zip(cfp_paths, mask_paths)):
        filename = os.path.basename(cfp_path)
        print(f"  [{i+1}/{len(cfp_paths)}] {filename}")

        # 1. 读取并处理CFP原图
        cfp_bgr = cv2.imread(cfp_path)
        if cfp_bgr is None:
            raise FileNotFoundError(f"无法读取CFP图像：{cfp_path}")
        cfp_rgb = cv2.cvtColor(cfp_bgr, cv2.COLOR_BGR2RGB)
        cfp_resized = cv2.resize(cfp_rgb, IMG_SIZE)

        # 保存CFP原图
        cv2.imwrite(
            os.path.join(out_cfp_dir, filename),
            cv2.cvtColor(cfp_resized, cv2.COLOR_RGB2BGR)
        )

        # 2. 生成伪OCTA：转灰度 + 直方图匹配OCTA风格
        cfp_gray = cv2.cvtColor(cfp_resized, cv2.COLOR_RGB2GRAY)
        pseudo_octa = match_histograms(cfp_gray, ref_hist)
        pseudo_octa = np.clip(pseudo_octa, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_octa_dir, filename), pseudo_octa)

        # 3. 处理血管金标准mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 兼容tif/gif等特殊格式标注
        if mask is None:
            mask_frames = imageio.mimread(mask_path)
            mask = np.array(mask_frames[0])
            # 强制转为单通道二维数组，避免维度异常
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        # 最近邻插值保证标签不模糊
        mask_resized = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        # 二值化
        mask_binary = np.zeros_like(mask_resized, dtype=np.uint8)
        mask_binary[mask_resized > 127] = 255
        cv2.imwrite(os.path.join(out_mask_dir, filename), mask_binary)

    print(f"✅ {cfp_split_name} 集处理完成")


def main():
    print("=" * 60)
    print("零训练生成伪配对多模态数据集（直方图匹配法）")
    print("=" * 60)

    # 1. 计算OCTA参考灰度分布
    print("\n1. 计算OCTA图像参考灰度分布...")
    octa_ref = get_octa_reference()
    print(f"   已加载 {REF_IMG_COUNT} 张OCTA参考图")

    # 2. 生成训练集
    process_split('training', 'train', octa_ref)

    # 3. 生成测试集
    process_split('test', 'test', octa_ref)

    print("\n" + "=" * 60)
    print(f"全部生成完成！数据集已保存到：{OUTPUT_ROOT}")
    print("\n生成目录结构：")
    print("pseudo_pair/")
    print("├── train/")
    print("│   ├── cfp/   # CFP彩色原图（256×256，与mask严格配对）")
    print("│   ├── octa/  # 伪OCTA灰度图（与CFP像素级对齐）")
    print("│   └── mask/  # 真实血管二值标注")
    print("└── test/")
    print("    ├── cfp/")
    print("    ├── octa/")
    print("    └── mask/")
    print("=" * 60)


if __name__ == "__main__":
    main()