import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两次卷积 + BN + ReLU，与单模态UNet结构完全一致"""
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
    """下采样：最大池化 + 两次卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样：转置卷积 + 跳跃连接 + 两次卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 尺寸自动对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出层：1x1卷积，输出原始logits，不含Sigmoid"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SEAttention(nn.Module):
    """通道注意力模块，用于特征融合时自适应加权"""
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


class UNet(nn.Module):
    """标准单模态UNet，与你现有单模态项目结构100%一致"""
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class EarlyFusionUNet(nn.Module):
    """早期融合基线：输入通道直接拼接，单流UNet"""
    def __init__(self, n_classes=1):
        super().__init__()
        # 3通道CFP + 1通道OCTA = 4通道输入
        self.unet = UNet(n_channels=4, n_classes=n_classes)

    def forward(self, x):
        return self.unet(x)


class AttentionFusionUNet(nn.Module):
    """
    特征级注意力融合模型：双流编码器 + 逐层注意力融合 + 共享解码器
    支持加载单模态预训练权重 + 冻结编码器主干
    【已修复】通道数严格对齐，无维度不匹配错误
    """
    def __init__(self, n_classes=1):
        super().__init__()
        base_ch = 64

        # ========== CFP编码器分支（3通道输入） ==========
        self.cfp_inc = DoubleConv(3, base_ch)
        self.cfp_down1 = Down(base_ch, base_ch * 2)
        self.cfp_down2 = Down(base_ch * 2, base_ch * 4)
        self.cfp_down3 = Down(base_ch * 4, base_ch * 8)

        # ========== OCTA编码器分支（1通道输入） ==========
        self.octa_inc = DoubleConv(1, base_ch)
        self.octa_down1 = Down(base_ch, base_ch * 2)
        self.octa_down2 = Down(base_ch * 2, base_ch * 4)
        self.octa_down3 = Down(base_ch * 4, base_ch * 8)

        # ========== 逐层注意力融合模块（拼接后通道翻倍） ==========
        self.att1 = SEAttention(base_ch * 2)   # f1: 64+64=128
        self.att2 = SEAttention(base_ch * 4)   # f2: 128+128=256
        self.att3 = SEAttention(base_ch * 8)   # f3: 256+256=512
        self.att4 = SEAttention(base_ch * 16)  # f4: 512+512=1024

        # ========== 瓶颈层（输入1024通道，输出2048通道） ==========
        self.bottleneck = Down(base_ch * 16, base_ch * 32)

        # ========== 共享解码器（严格对齐跳跃连接通道数） ==========
        self.up1 = Up(base_ch * 32, base_ch * 16)  # 2048上采样→1024 + f4(1024) → 输出1024
        self.up2 = Up(base_ch * 16, base_ch * 8)   # 1024上采样→512  + f3(512)  → 输出512
        self.up3 = Up(base_ch * 8, base_ch * 4)    # 512上采样→256   + f2(256)  → 输出256
        self.up4 = Up(base_ch * 4, base_ch * 2)    # 256上采样→128   + f1(128)  → 输出128

        # 输出层
        self.outc = OutConv(base_ch * 2, n_classes)

    def forward(self, x):
        # 拆分输入：前3通道CFP，第4通道OCTA
        x_cfp = x[:, :3, :, :]
        x_octa = x[:, 3:, :, :]

        # 双流编码 + 逐层注意力融合
        c1 = self.cfp_inc(x_cfp)
        o1 = self.octa_inc(x_octa)
        f1 = torch.cat([c1, o1], dim=1)
        f1 = self.att1(f1)

        c2 = self.cfp_down1(c1)
        o2 = self.octa_down1(o1)
        f2 = torch.cat([c2, o2], dim=1)
        f2 = self.att2(f2)

        c3 = self.cfp_down2(c2)
        o3 = self.octa_down2(o2)
        f3 = torch.cat([c3, o3], dim=1)
        f3 = self.att3(f3)

        c4 = self.cfp_down3(c3)
        o4 = self.octa_down3(o3)
        f4 = torch.cat([c4, o4], dim=1)
        f4 = self.att4(f4)

        # 瓶颈层
        bottleneck = self.bottleneck(f4)

        # 解码器 + 跳跃连接
        x = self.up1(bottleneck, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        return self.outc(x)