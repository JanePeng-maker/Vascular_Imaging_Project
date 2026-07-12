## 标准运行步骤
1. 先进入 code 目录
bash
cd "/Desktop/眼底血管项目/MultiModal_Vessel_Project/code”（更换成运行者自己的路径）

2. 生成伪配对数据集
bash
python generate_pseudo_pair.py

3. 放置预训练权重（重命名对应文件名）
CFP 权重 → ../data/cfp_pretrained/cfp_unet_best.pth
OCTA 权重 → ../data/octa_pretrained/octa_unet_best.pth

4. 运行训练 + 测试
bash
python main.py