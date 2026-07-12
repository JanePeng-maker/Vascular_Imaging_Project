import os
import cv2
import numpy as np
from ultralytics import YOLO
import tifffile
# ==================== 全局变量 ====================
# 自动获取当前文件所在目录作为项目根
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型路径：默认放在项目根目录，命名为best.pt；也可自行修改为runs下的训练权重
MODEL_PATH = os.path.join(ROOT_DIR, "best.pt")
INPUT_DIR = os.path.join(ROOT_DIR, "Originaldata", "Testing Set")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
DEVICE = "cpu"
# ==================================================
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
def load_image(image_path):
    """支持 jpg/png/tif 多格式读取"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        img = tifffile.imread(image_path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path)
    return img
def predict_one(model, image_path, output_dir):
    original_img = load_image(image_path)
    if original_img is None:
        return None
    h, w = original_img.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    results = model.predict(
        source=original_img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False,
        retina_masks=True,
    )
    # 生成合并的二值掩码
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    result = results[0]
    if result.masks is not None:
        for mask in result.masks.data.cpu().numpy():
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (mask_resized > 0.5).astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, binary)
    # 保存 tif 格式掩码
    out_tif = os.path.join(output_dir, f"{base_name}_HE_result.tif")
    tifffile.imwrite(out_tif, combined_mask)
    # 保存可视化叠加图
    vis_img = original_img.copy()
    if result.masks is not None and len(result.masks) > 0:
        mask_red = np.zeros_like(original_img)
        mask_red[combined_mask > 0] = [0, 0, 255]
        vis_img = cv2.addWeighted(vis_img, 0.7, mask_red, 0.5, 0)
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"HE:{conf:.2f}", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_visual.jpg"), vis_img)
    return out_tif
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    # 收集待检测图片
    image_files = []
    if os.path.isdir(INPUT_DIR):
        for f in sorted(os.listdir(INPUT_DIR)):
            if f.lower().endswith(SUPPORTED_FORMATS):
                image_files.append(os.path.join(INPUT_DIR, f))
    else:
        if INPUT_DIR.lower().endswith(SUPPORTED_FORMATS):
            image_files.append(INPUT_DIR)
    print(f"找到 {len(image_files)} 张待检测图像\n")
    ok = 0
    for i, p in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {os.path.basename(p)}")
        if predict_one(model, p, OUTPUT_DIR):
            ok += 1
    print(f"\n完成！成功 {ok}/{len(image_files)}")
    print(f"结果目录: {OUTPUT_DIR}")
if __name__ == "__main__":
    main()
