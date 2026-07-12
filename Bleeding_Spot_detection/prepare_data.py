import os
import cv2
import numpy as np
# ==================== 全局变量 ====================
# 自动获取当前文件所在目录作为项目根
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_CONTOUR_POINTS = 3
# ==================================================
def mask_to_yolo_labels(mask_path, img_w, img_h, class_id=0):
    """IDRiD红色掩码 -> YOLO分割标签（归一化多边形坐标）"""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []
    # 提取红色通道（IDRiD的HE出血点是红色标注）
    if len(mask.shape) == 3:
        mask_gray = mask[:, :, 2]
    else:
        mask_gray = mask
    _, binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for cnt in contours:
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < MIN_CONTOUR_POINTS:
            continue
        points = approx.reshape(-1, 2).astype(np.float32)
        points[:, 0] /= img_w
        points[:, 1] /= img_h
        points = np.clip(points, 0.0, 1.0)
        label_str = f"{class_id}"
        for x, y in points:
            label_str += f" {x:.6f} {y:.6f}"
        labels.append(label_str)
    return labels
def process_split(original_dir, gt_dir, img_out_dir, lbl_out_dir, split_name):
    count = 0
    for img_name in sorted(os.listdir(original_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_HE.tif"
        mask_path = os.path.join(gt_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"  跳过 {img_name}: 无对应掩码")
            continue
        img_path = os.path.join(original_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        labels = mask_to_yolo_labels(mask_path, w, h)
        if not labels:
            print(f"  警告: {img_name} 无有效出血区域")
            continue
        cv2.imwrite(os.path.join(img_out_dir, f"{base_name}.jpg"), img)
        with open(os.path.join(lbl_out_dir, f"{base_name}.txt"), 'w') as f:
            f.write('\n'.join(labels))
        count += 1
    print(f"  {split_name}: {count} 张")
    return count
def main():
    original_train = os.path.join(ROOT_DIR, "Originaldata", "Training Set")
    original_test = os.path.join(ROOT_DIR, "Originaldata", "Testing Set")
    gt_train = os.path.join(ROOT_DIR, "GT", "Training Set")
    gt_test = os.path.join(ROOT_DIR, "GT", "Testing Set")
    dataset_dir = os.path.join(ROOT_DIR, "dataset")
    dirs = [
        os.path.join(dataset_dir, "images", "train"),
        os.path.join(dataset_dir, "images", "val"),
        os.path.join(dataset_dir, "labels", "train"),
        os.path.join(dataset_dir, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("=" * 40)
    print("数据预处理开始（使用官方划分）")
    print("=" * 40)
    train_n = process_split(original_train, gt_train, dirs[0], dirs[2], "训练集")
    val_n = process_split(original_test, gt_test, dirs[1], dirs[3], "验证集")
    print("=" * 40)
    print(f"完成！训练集 {train_n} 张，验证集 {val_n} 张")
    print(f"输出目录: {dataset_dir}")
if __name__ == "__main__":
    main()
