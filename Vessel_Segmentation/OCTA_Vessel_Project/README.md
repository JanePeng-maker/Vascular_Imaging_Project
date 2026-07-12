### 运行程序步骤
1.  在 CMD 中进入项目文件夹：
    
    bash
    运行
    
    ```
    cd C:\Users\你的用户名\Desktop\OCTA_Vessel_Project
    ```
    
2.  执行运行命令：
    
    bash
    运行
    
    ```
    python main.py
    ```
### 查看结果

-   训练好的模型和损失曲线：`OCTA_Vessel_Project/saved_model/`
-   测试分割对比图：`OCTA_Vessel_Project/test_results/`

* * *

后续可调参数说明
--------

所有参数都在`main.py`最开头，直接修改数字即可：

-   修改训练样本数：改 `N_TRAIN_SAMPLES`
-   修改测试样本数：改 `N_TEST_SAMPLES`
-   增加训练轮次：改 `EPOCHS`
-   调整血管权重：改 `POS_WEIGHT`（血管越细越少，数值可以调大到 15~20）
-   缩小图片加速训练：改 `IMG_SIZE`，比如改成 `(256, 256)`