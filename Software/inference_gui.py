import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 资源路径兼容（开发环境 + PyInstaller 打包）
# ============================================================
def resource_path(relative_path):
    if getattr(sys, '_MEIPASS', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# ============================================================
# 【血管分割模型：AttentionFusionUNet】
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionFusionUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        base_ch = 64
        self.cfp_inc = DoubleConv(3, base_ch)
        self.cfp_down1 = Down(base_ch, base_ch * 2)
        self.cfp_down2 = Down(base_ch * 2, base_ch * 4)
        self.cfp_down3 = Down(base_ch * 4, base_ch * 8)
        self.octa_inc = DoubleConv(1, base_ch)
        self.octa_down1 = Down(base_ch, base_ch * 2)
        self.octa_down2 = Down(base_ch * 2, base_ch * 4)
        self.octa_down3 = Down(base_ch * 4, base_ch * 8)
        self.att1 = SEAttention(base_ch * 2)
        self.att2 = SEAttention(base_ch * 4)
        self.att3 = SEAttention(base_ch * 8)
        self.att4 = SEAttention(base_ch * 16)
        self.bottleneck = Down(base_ch * 16, base_ch * 32)
        self.up1 = Up(base_ch * 32, base_ch * 16)
        self.up2 = Up(base_ch * 16, base_ch * 8)
        self.up3 = Up(base_ch * 8, base_ch * 4)
        self.up4 = Up(base_ch * 4, base_ch * 2)
        self.outc = OutConv(base_ch * 2, n_classes)

    def forward(self, x):
        x_cfp = x[:, :3, :, :]
        x_octa = x[:, 3:, :, :]
        c1 = self.cfp_inc(x_cfp)
        o1 = self.octa_inc(x_octa)
        f1 = self.att1(torch.cat([c1, o1], dim=1))
        c2 = self.cfp_down1(c1)
        o2 = self.octa_down1(o1)
        f2 = self.att2(torch.cat([c2, o2], dim=1))
        c3 = self.cfp_down2(c2)
        o3 = self.octa_down2(o2)
        f3 = self.att3(torch.cat([c3, o3], dim=1))
        c4 = self.cfp_down3(c3)
        o4 = self.octa_down3(o3)
        f4 = self.att4(torch.cat([c4, o4], dim=1))
        bottleneck = self.bottleneck(f4)
        x = self.up1(bottleneck, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        return self.outc(x)


# ============================================================
# 全局配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VESSEL_IMG_SIZE = (256, 256)
VESSEL_MEAN, VESSEL_STD = 0.5, 0.5
VESSEL_THRESHOLD = 0.5
BLEED_CONF = 0.25
BLEED_IOU = 0.45
BLEED_IMGSZ = 640

SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
OUTPUT_FORMATS = ["tif", "png", "jpg", "bmp"]

vessel_model_path = resource_path("att_fusion_best.pth")
bleed_model_path = resource_path("best.pt")

vessel_model = None
bleed_model = None


# ============================================================
# 工具函数：全格式图像读写
# ============================================================
def load_image_any(path, force_gray=False):
    """
    读取任意格式2D图像，返回 uint8 numpy
    force_gray=True 返回 (H,W) 灰度图，否则返回 (H,W,3) BGR 图
    支持 8位/16位/32位 灰度、RGB、RGBA，tif/png/jpg/bmp
    多层 fallback：tifffile → cv2 → PIL
    """
    ext = os.path.splitext(path)[1].lower()
    img = None

    # ---- 第1层：tif/tiff 优先用 tifffile ----
    if ext in ('.tif', '.tiff'):
        try:
            import tifffile
            img = tifffile.imread(path)
        except Exception:
            img = None

    # ---- 第2层：OpenCV 读取（非tif直接走这里；tif失败也走这里兜底）----
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # ---- 第3层：PIL 兜底 ----
    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(path)
            img = np.array(pil_img)
        except Exception:
            pass

    if img is None:
        raise ValueError(f"无法读取图像: {path}（尝试了 tifffile / cv2 / PIL 均失败）")

    # ---- 归一化到 uint8 [0,255] ----
    if img.dtype == np.uint16:
        max_val = img.max()
        if max_val > 0:
            img = (img.astype(np.float32) / max_val * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    elif img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # ---- 通道处理：先算出灰度图 ----
    if len(img.shape) == 2:
        gray = img
    elif img.shape[2] == 4:
        # RGBA → 灰度
        if ext in ('.tif', '.tiff') and img.dtype == np.uint8:
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.shape[2] == 3:
        if ext in ('.tif', '.tiff'):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.squeeze()

    if force_gray:
        return gray

    # ---- 转 BGR 三通道 ----
    if len(img.shape) == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        if ext in ('.tif', '.tiff'):
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 3:
        if ext in ('.tif', '.tiff'):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    return img


def save_mask(mask_uint8, save_path, fmt='tif'):
    """保存二值掩码，支持 tif/png/jpg/bmp"""
    fmt = fmt.lower()
    if fmt in ('tif', 'tiff'):
        tifffile.imwrite(save_path, mask_uint8)
    elif fmt == 'jpg' or fmt == 'jpeg':
        # jpg 有损，二值图用高质量参数
        cv2.imwrite(save_path, mask_uint8, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(save_path, mask_uint8)


def list_images(folder):
    files = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(SUPPORTED_EXT):
            files.append(os.path.join(folder, f))
    return files


# ============================================================
# 血管分割推理（支持单模态 / 双模态）
# ============================================================
def load_vessel_model():
    global vessel_model
    if not os.path.exists(vessel_model_path):
        return False, f"未找到血管分割模型: {vessel_model_path}"
    try:
        vessel_model = AttentionFusionUNet(n_classes=1).to(DEVICE)
        state = torch.load(vessel_model_path, map_location=DEVICE, weights_only=False)
        vessel_model.load_state_dict(state)
        vessel_model.eval()
        return True, "血管分割模型加载成功"
    except Exception as e:
        vessel_model = None
        return False, f"血管分割模型加载失败: {str(e)}"


def vessel_predict_single(cfp_path=None, octa_path=None, save_path=None, output_fmt='tif'):
    """
    血管分割推理，支持三种模式：
    1. 双模态（CFP+OCTA都有）→ 正常多模态融合（效果最好）
    2. 仅CFP → OCTA通道由CFP灰度图填充（降级模式）
    3. 仅OCTA → CFP通道由OCTA复制3通道填充（降级模式）
    """
    if cfp_path is None and octa_path is None:
        raise ValueError("CFP和OCTA至少提供一个")

    # ---- 读取并获取原图尺寸 ----
    ref_path = cfp_path if cfp_path else octa_path
    ref_bgr = load_image_any(ref_path)
    orig_h, orig_w = ref_bgr.shape[:2]

    # ---- 构造 CFP 3通道张量 ----
    if cfp_path is not None:
        cfp_bgr = load_image_any(cfp_path)
    else:
        # 只有 OCTA，复制3遍当 CFP 用
        octa_gray = load_image_any(octa_path, force_gray=True)
        cfp_bgr = cv2.cvtColor(octa_gray, cv2.COLOR_GRAY2BGR)

    cfp_rgb = cv2.cvtColor(cfp_bgr, cv2.COLOR_BGR2RGB)
    cfp = cv2.resize(cfp_rgb, VESSEL_IMG_SIZE).astype(np.float32) / 255.0
    cfp = (cfp - VESSEL_MEAN) / VESSEL_STD
    cfp_t = torch.from_numpy(cfp).permute(2, 0, 1).float()

    # ---- 构造 OCTA 单通道张量 ----
    if octa_path is not None:
        octa_gray = load_image_any(octa_path, force_gray=True)
    else:
        # 只有 CFP，取灰度当 OCTA 用
        octa_gray = cv2.cvtColor(cfp_bgr, cv2.COLOR_BGR2GRAY)

    octa = cv2.resize(octa_gray, VESSEL_IMG_SIZE).astype(np.float32) / 255.0
    octa = (octa - VESSEL_MEAN) / VESSEL_STD
    octa_t = torch.from_numpy(octa).unsqueeze(0).float()

    # ---- 拼接 4 通道输入 ----
    inp = torch.cat([cfp_t, octa_t], dim=0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = vessel_model(inp)
        pred = torch.sigmoid(out).squeeze().cpu().numpy()

    pred_binary = (pred > VESSEL_THRESHOLD).astype(np.uint8) * 255
    pred_full = cv2.resize(pred_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 带扩展名保存
    if not save_path.endswith(f".{output_fmt}"):
        save_path = f"{os.path.splitext(save_path)[0]}.{output_fmt}"
    save_mask(pred_full, save_path, output_fmt)
    return True


# ============================================================
# 出血点检测推理（YOLOv8-Seg，全格式输入 + 可选输出格式）
# ============================================================
def load_bleed_model():
    global bleed_model
    if not os.path.exists(bleed_model_path):
        return False, f"未找到出血点模型: {bleed_model_path}"
    try:
        from ultralytics import YOLO
        bleed_model = YOLO(bleed_model_path)
        return True, "出血点检测模型加载成功"
    except Exception as e:
        bleed_model = None
        return False, f"出血点模型加载失败: {str(e)}"


def bleed_predict_single(img_path, save_path, output_fmt='tif'):
    img = load_image_any(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]

    results = bleed_model.predict(
        source=img, conf=BLEED_CONF, iou=BLEED_IOU,
        imgsz=BLEED_IMGSZ,
        device=DEVICE.type if DEVICE.type != 'cpu' else 'cpu',
        verbose=False, retina_masks=True,
    )
    combined = np.zeros((h, w), dtype=np.uint8)
    r = results[0]
    if r.masks is not None:
        for mask in r.masks.data.cpu().numpy():
            m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (m > 0.5).astype(np.uint8) * 255
            combined = np.maximum(combined, binary)

    if not save_path.endswith(f".{output_fmt}"):
        save_path = f"{os.path.splitext(save_path)[0]}.{output_fmt}"
    save_mask(combined, save_path, output_fmt)
    return True


# ============================================================
# GUI 主程序
# ============================================================
class MedicalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("医学影像分析工具 - 血管分割 & 出血点检测")
        self.root.geometry("1380x760")
        self.root.resizable(False, False)

        self.bleed_input = tk.StringVar()
        self.bleed_output = tk.StringVar()
        self.bleed_fmt = tk.StringVar(value="tif")

        self.vessel_cfp = tk.StringVar()
        self.vessel_octa = tk.StringVar()
        self.vessel_output = tk.StringVar()
        self.vessel_fmt = tk.StringVar(value="tif")

        self.status_text = tk.StringVar(value="就绪")
        self.progress_val = tk.DoubleVar(value=0)
        self.running = False
        self.stop_flag = False

        self._build_ui()
        self._auto_load_models()

    def _build_ui(self):
        ENTRY_WIDTH = 58

        # ========== 出血点检测区 ==========
        bleed_frame = tk.LabelFrame(self.root, text=" 出血点检测（单模态 CFP） ",
                                    padx=12, pady=10, fg="#d2691e")
        bleed_frame.pack(fill='x', padx=20, pady=(12, 6))

        tk.Label(bleed_frame, text="输入（文件 / 文件夹）:").grid(row=0, column=0, sticky='e', padx=4, pady=4)
        tk.Entry(bleed_frame, textvariable=self.bleed_input, width=ENTRY_WIDTH).grid(row=0, column=1, padx=4)
        tk.Button(bleed_frame, text="选择文件", width=10, command=self._pick_bleed_file).grid(row=0, column=2, padx=2)
        tk.Button(bleed_frame, text="选择文件夹", width=10, command=self._pick_bleed_folder).grid(row=0, column=3, padx=2)

        tk.Label(bleed_frame, text="输出文件夹:").grid(row=1, column=0, sticky='e', padx=4, pady=4)
        tk.Entry(bleed_frame, textvariable=self.bleed_output, width=ENTRY_WIDTH).grid(row=1, column=1, padx=4)
        tk.Button(bleed_frame, text="浏览", width=10, command=self._pick_bleed_out).grid(row=1, column=2, padx=2)

        tk.Label(bleed_frame, text="输出格式:").grid(row=2, column=0, sticky='e', padx=4, pady=4)
        ttk.Combobox(bleed_frame, textvariable=self.bleed_fmt, values=OUTPUT_FORMATS,
                     state="readonly", width=10).grid(row=2, column=1, sticky='w', padx=4)

        self.btn_bleed_start = tk.Button(bleed_frame, text="▶ 开始\n出血点检测",
                                         bg="#e67e22", fg="white",
                                         width=14, height=2, font=("Arial", 11, "bold"),
                                         command=self._run_bleed)
        self.btn_bleed_start.grid(row=0, column=4, rowspan=3, padx=(18, 4), ipadx=4)

        # ========== 血管分割区 ==========
        vessel_frame = tk.LabelFrame(self.root, text=" 血管分割（支持 CFP / OCTA / 双模态 三种模式） ",
                                     padx=12, pady=10, fg="#2e8b57")
        vessel_frame.pack(fill='x', padx=20, pady=6)

        tk.Label(vessel_frame, text="CFP（可选）:").grid(row=0, column=0, sticky='e', padx=4, pady=4)
        tk.Entry(vessel_frame, textvariable=self.vessel_cfp, width=ENTRY_WIDTH).grid(row=0, column=1, padx=4)
        tk.Button(vessel_frame, text="选择文件", width=10, command=self._pick_cfp_file).grid(row=0, column=2, padx=2)
        tk.Button(vessel_frame, text="选择文件夹", width=10, command=self._pick_cfp_folder).grid(row=0, column=3, padx=2)

        tk.Label(vessel_frame, text="OCTA（可选）:").grid(row=1, column=0, sticky='e', padx=4, pady=4)
        tk.Entry(vessel_frame, textvariable=self.vessel_octa, width=ENTRY_WIDTH).grid(row=1, column=1, padx=4)
        tk.Button(vessel_frame, text="选择文件", width=10, command=self._pick_octa_file).grid(row=1, column=2, padx=2)
        tk.Button(vessel_frame, text="选择文件夹", width=10, command=self._pick_octa_folder).grid(row=1, column=3, padx=2)

        tk.Label(vessel_frame, text="输出文件夹:").grid(row=2, column=0, sticky='e', padx=4, pady=4)
        tk.Entry(vessel_frame, textvariable=self.vessel_output, width=ENTRY_WIDTH).grid(row=2, column=1, padx=4)
        tk.Button(vessel_frame, text="浏览", width=10, command=self._pick_vessel_out).grid(row=2, column=2, padx=2)

        tk.Label(vessel_frame, text="输出格式:").grid(row=3, column=0, sticky='e', padx=4, pady=4)
        ttk.Combobox(vessel_frame, textvariable=self.vessel_fmt, values=OUTPUT_FORMATS,
                     state="readonly", width=10).grid(row=3, column=1, sticky='w', padx=4)
        tk.Label(vessel_frame, text="※ CFP和OCTA至少填一个；都有时效果最佳", fg="#888").grid(row=3, column=2, columnspan=2, sticky='w')

        self.btn_vessel_start = tk.Button(vessel_frame, text="▶ 开始\n血管分割",
                                          bg="#27ae60", fg="white",
                                          width=14, height=2, font=("Arial", 11, "bold"),
                                          command=self._run_vessel)
        self.btn_vessel_start.grid(row=0, column=4, rowspan=4, padx=(18, 4), ipadx=4)

        # ========== 控制区：停止按钮 ==========
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(fill='x', padx=20, pady=(10, 4))
        self.btn_stop = tk.Button(ctrl_frame, text="⏹ 停止处理",
                                  bg="#c0392b", fg="white",
                                  width=16, font=("Arial", 10, "bold"),
                                  command=self._stop_process, state='disabled')
        self.btn_stop.pack(side='right')

        # ========== 进度条 ==========
        prog_frame = tk.Frame(self.root)
        prog_frame.pack(fill='x', padx=20, pady=6)
        tk.Label(prog_frame, text="进度:").pack(side='left')
        self.progress = ttk.Progressbar(prog_frame, variable=self.progress_val,
                                        maximum=100, length=1050)
        self.progress.pack(side='left', padx=10, fill='x', expand=True)

        # ========== 状态栏 ==========
        tk.Label(self.root, textvariable=self.status_text, fg="gray",
                 wraplength=1300, justify='left').pack(pady=6)

        # ========== 模型路径设置区 ==========
        model_frame = tk.LabelFrame(self.root, text=" 模型文件路径 ",
                                    padx=12, pady=8, fg="#666")
        model_frame.pack(fill='x', padx=20, pady=(6, 12))

        tk.Label(model_frame, text="出血点模型 (best.pt):").grid(row=0, column=0, sticky='e', padx=4, pady=3)
        self.bleed_model_entry = tk.Entry(model_frame, width=80)
        self.bleed_model_entry.insert(0, bleed_model_path)
        self.bleed_model_entry.grid(row=0, column=1, padx=4)
        tk.Button(model_frame, text="选择", width=8, command=self._pick_bleed_model).grid(row=0, column=2, padx=4)

        tk.Label(model_frame, text="血管分割模型 (att_fusion_best.pth):").grid(row=1, column=0, sticky='e', padx=4, pady=3)
        self.vessel_model_entry = tk.Entry(model_frame, width=80)
        self.vessel_model_entry.insert(0, vessel_model_path)
        self.vessel_model_entry.grid(row=1, column=1, padx=4)
        tk.Button(model_frame, text="选择", width=8, command=self._pick_vessel_model).grid(row=1, column=2, padx=4)

        tk.Button(model_frame, text="重新加载模型", width=14,
                  command=self._reload_models).grid(row=0, column=3, rowspan=2, padx=10)

    # ---- 文件选择 ----
    def _pick_bleed_file(self):
        p = filedialog.askopenfilename(title="选择图像", filetypes=[("所有图像", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
        if p: self.bleed_input.set(p)
    def _pick_bleed_folder(self):
        p = filedialog.askdirectory(title="选择图像文件夹")
        if p: self.bleed_input.set(p)
    def _pick_bleed_out(self):
        p = filedialog.askdirectory(title="选择输出文件夹")
        if p: self.bleed_output.set(p)

    def _pick_cfp_file(self):
        p = filedialog.askopenfilename(title="选择CFP图像")
        if p: self.vessel_cfp.set(p)
    def _pick_cfp_folder(self):
        p = filedialog.askdirectory(title="选择CFP文件夹")
        if p: self.vessel_cfp.set(p)
    def _pick_octa_file(self):
        p = filedialog.askopenfilename(title="选择OCTA图像")
        if p: self.vessel_octa.set(p)
    def _pick_octa_folder(self):
        p = filedialog.askdirectory(title="选择OCTA文件夹")
        if p: self.vessel_octa.set(p)
    def _pick_vessel_out(self):
        p = filedialog.askdirectory(title="选择输出文件夹")
        if p: self.vessel_output.set(p)

    def _pick_bleed_model(self):
        p = filedialog.askopenfilename(title="选择出血点模型", filetypes=[("PyTorch", "*.pt *.pth")])
        if p:
            self.bleed_model_entry.delete(0, tk.END)
            self.bleed_model_entry.insert(0, p)
    def _pick_vessel_model(self):
        p = filedialog.askopenfilename(title="选择血管分割模型", filetypes=[("PyTorch", "*.pt *.pth")])
        if p:
            self.vessel_model_entry.delete(0, tk.END)
            self.vessel_model_entry.insert(0, p)

    # ---- 模型加载 ----
    def _auto_load_models(self):
        msgs = []
        _, msg1 = load_vessel_model()
        msgs.append(msg1)
        _, msg2 = load_bleed_model()
        msgs.append(msg2)
        self.status_text.set(" | ".join(msgs))

    def _reload_models(self):
        global vessel_model_path, bleed_model_path
        vessel_model_path = self.vessel_model_entry.get().strip()
        bleed_model_path = self.bleed_model_entry.get().strip()
        self._auto_load_models()

    # ---- 按钮状态 ----
    def _set_running_state(self, running):
        state = 'disabled' if running else 'normal'
        self.btn_bleed_start.config(state=state)
        self.btn_vessel_start.config(state=state)
        self.btn_stop.config(state='normal' if running else 'disabled')

    def _stop_process(self):
        if self.running:
            self.stop_flag = True
            self._set_status("⏹ 正在停止（当前图片处理完后中止）...", "orange")

    # ---- 出血点检测线程 ----
    def _run_bleed(self):
        if self.running: return
        if bleed_model is None:
            messagebox.showerror("错误", "出血点模型未加载，请检查模型路径")
            return
        inp = self.bleed_input.get().strip()
        out = self.bleed_output.get().strip()
        fmt = self.bleed_fmt.get().strip()
        if not inp or not out:
            messagebox.showwarning("警告", "请填写输入和输出路径")
            return
        if not os.path.exists(inp):
            messagebox.showerror("错误", "输入路径不存在")
            return
        os.makedirs(out, exist_ok=True)
        self.running = True
        self.stop_flag = False
        self._set_running_state(True)
        threading.Thread(target=self._bleed_worker, args=(inp, out, fmt), daemon=True).start()

    def _bleed_worker(self, inp, out, fmt):
        try:
            files = [inp] if os.path.isfile(inp) else list_images(inp)
            total = len(files)
            if total == 0:
                self._set_status("未找到图像文件", "red"); return
            ok = 0
            for i, f in enumerate(files):
                if self.stop_flag:
                    self._set_status(f"⏹ 已停止（完成 {i}/{total} 张）", "orange"); return
                base = os.path.splitext(os.path.basename(f))[0]
                try:
                    bleed_predict_single(f, os.path.join(out, f"{base}_HE_result.{fmt}"), fmt)
                    ok += 1
                except Exception as e:
                    print(f"{f} 失败: {e}")
                self._set_progress((i + 1) / total * 100)
                self._set_status(f"出血点检测: {i+1}/{total}  {os.path.basename(f)}")
            self._set_status(f"✅ 出血点检测完成！成功 {ok}/{total}，结果: {out}", "green")
        except Exception as e:
            self._set_status(f"❌ 出错: {str(e)}", "red")
        finally:
            self.running = False
            self.stop_flag = False
            self._set_running_state(False)

    # ---- 血管分割线程 ----
    def _run_vessel(self):
        if self.running: return
        if vessel_model is None:
            messagebox.showerror("错误", "血管分割模型未加载，请检查模型路径")
            return
        cfp_p = self.vessel_cfp.get().strip()
        octa_p = self.vessel_octa.get().strip()
        out = self.vessel_output.get().strip()
        fmt = self.vessel_fmt.get().strip()

        if not cfp_p and not octa_p:
            messagebox.showwarning("警告", "CFP和OCTA至少填写一个")
            return
        if not out:
            messagebox.showwarning("警告", "请填写输出文件夹")
            return
        if cfp_p and not os.path.exists(cfp_p):
            messagebox.showerror("错误", f"CFP路径不存在: {cfp_p}")
            return
        if octa_p and not os.path.exists(octa_p):
            messagebox.showerror("错误", f"OCTA路径不存在: {octa_p}")
            return

        os.makedirs(out, exist_ok=True)
        self.running = True
        self.stop_flag = False
        self._set_running_state(True)
        threading.Thread(target=self._vessel_worker, args=(cfp_p, octa_p, out, fmt), daemon=True).start()

    def _vessel_worker(self, cfp_p, octa_p, out, fmt):
        try:
            # ---- 判断模式 ----
            cfp_is_file = cfp_p and os.path.isfile(cfp_p)
            cfp_is_dir = cfp_p and os.path.isdir(cfp_p)
            octa_is_file = octa_p and os.path.isfile(octa_p)
            octa_is_dir = octa_p and os.path.isdir(octa_p)

            pairs = []  # 每个元素: (cfp_path or None, octa_path or None, basename)

            if cfp_is_file and octa_is_file:
                # 双文件模式
                base = os.path.splitext(os.path.basename(cfp_p))[0]
                pairs = [(cfp_p, octa_p, base)]

            elif cfp_is_dir and octa_is_dir:
                # 双文件夹：按文件名配对
                cfp_map = {os.path.splitext(os.path.basename(f))[0]: f for f in list_images(cfp_p)}
                octa_map = {os.path.splitext(os.path.basename(f))[0]: f for f in list_images(octa_p)}
                common = sorted(set(cfp_map.keys()) & set(octa_map.keys()))
                pairs = [(cfp_map[k], octa_map[k], k) for k in common]
                if not pairs:
                    self._set_status("CFP和OCTA文件夹中未找到文件名匹配的图像对", "red"); return

            elif cfp_is_file and not octa_p:
                # 仅 CFP 文件
                base = os.path.splitext(os.path.basename(cfp_p))[0]
                pairs = [(cfp_p, None, base)]

            elif cfp_is_dir and not octa_p:
                # 仅 CFP 文件夹
                cfp_files = list_images(cfp_p)
                pairs = [(f, None, os.path.splitext(os.path.basename(f))[0]) for f in cfp_files]

            elif octa_is_file and not cfp_p:
                # 仅 OCTA 文件
                base = os.path.splitext(os.path.basename(octa_p))[0]
                pairs = [(None, octa_p, base)]

            elif octa_is_dir and not cfp_p:
                # 仅 OCTA 文件夹
                octa_files = list_images(octa_p)
                pairs = [(None, f, os.path.splitext(os.path.basename(f))[0]) for f in octa_files]

            else:
                self._set_status("CFP和OCTA必须同为文件或同为文件夹（或只填一个）", "red"); return

            total = len(pairs)
            if total == 0:
                self._set_status("未找到图像文件", "red"); return

            ok = 0
            for i, (cf, of, base) in enumerate(pairs):
                if self.stop_flag:
                    self._set_status(f"⏹ 已停止（完成 {i}/{total} 张）", "orange"); return
                try:
                    vessel_predict_single(
                        cfp_path=cf, octa_path=of,
                        save_path=os.path.join(out, f"{base}_vessel_result.{fmt}"),
                        output_fmt=fmt
                    )
                    ok += 1
                except Exception as e:
                    print(f"{base} 失败: {e}")
                self._set_progress((i + 1) / total * 100)
                mode_str = "双模态" if cf and of else ("仅CFP" if cf else "仅OCTA")
                self._set_status(f"血管分割[{mode_str}]: {i+1}/{total}  {base}")

            self._set_status(f"✅ 血管分割完成！成功 {ok}/{total}，结果: {out}", "green")
        except Exception as e:
            self._set_status(f"❌ 出错: {str(e)}", "red")
        finally:
            self.running = False
            self.stop_flag = False
            self._set_running_state(False)

    # ---- UI 更新 ----
    def _set_status(self, text, color="gray"):
        self.root.after(0, lambda: self.status_text.set(text))

    def _set_progress(self, val):
        self.root.after(0, lambda: self.progress_val.set(val))


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalApp(root)
    root.mainloop()
