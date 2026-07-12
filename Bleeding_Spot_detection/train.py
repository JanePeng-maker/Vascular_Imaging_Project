import os
from ultralytics import YOLO

# 自动获取当前脚本所在目录作为项目根目录
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(SCRIPT_PATH)

# ==================== 全局超参（路径全部自动基于ROOT，无需手动改） ====================
EPOCHS = 10            # 最大训练轮数（带早停，实际100-200轮收敛）
IMG_SIZE = 640          # 输入尺寸
BATCH_SIZE = 8          # 显存不够改4
MODEL_TYPE = "yolov8s-seg.pt"
PATIENCE = 50           # 早停：连续50轮无提升自动停止
DEVICE = "cpu"            # GPU填数字0/1，CPU写 "cpu"
# 配置文件自动拼接完整路径
DATA_YAML_PATH = os.path.join(ROOT_DIR, "config.yaml")
# 训练输出目录（相对项目根）
SAVE_PROJECT_DIR = os.path.join(ROOT_DIR, "runs/segment")
TRAIN_NAME = "train"
# ==================================================

def main():
    # 切换工作目录到当前项目根
    os.chdir(ROOT_DIR)

    print(f"项目根目录：{ROOT_DIR}")
    print(f"训练配置文件路径：{DATA_YAML_PATH}")
    print(f"加载模型: {MODEL_TYPE}")
    model = YOLO(MODEL_TYPE)

    print(f"开始训练，最大 {EPOCHS} 轮，早停耐心值 {PATIENCE}")
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        # 医学影像专用增强（颜色改动小，几何增强多）
        hsv_h=0.0,
        hsv_s=0.1,
        hsv_v=0.1,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # 优化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # 保存
        save=True,
        save_period=10,
        plots=True,
        project=SAVE_PROJECT_DIR,
        name=TRAIN_NAME,
        seed=42,
        verbose=True,
    )

    best_weight_path = os.path.join(SAVE_PROJECT_DIR, TRAIN_NAME, "weights/best.pt")
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳权重完整路径: {best_weight_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
