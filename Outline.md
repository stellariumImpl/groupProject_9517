## 基於現有的模型，數據集 做了什麽（more than existing code）

**使用了V-01 V-02數據集叠加**

### data_split

使用csv文件存儲圖片（編號）

時間序列分割數據集，按照文件名（UNIX時間戳）排序，按照7:2:1分割成train valid test

建立像素灰度值與trainId的映射，對讀入的indexLabel圖片（灰度圖），將像素點映射到灰度值對應的trainId

![image-20240714225750625](https://gcore.jsdelivr.net/gh/stellariumImpl/CDN/pic/image-20240714225750625.png)

### data_load

![image-20240715005750793](https://gcore.jsdelivr.net/gh/stellariumImpl/CDN/pic/image-20240715005750793.png)

提供從trainId到rgb的映射方法：建立trainId與rgb色的映射，將data_split中trainId標注的indexLabel圖片轉換成rgb色

根據dataset_type（訓練集，驗證集，測試集），提供了不同的方式的數據增强組合和預處理操作：

- 訓練階段：

  1. 圖像大小調整（Resize），將輸入image和對應label調整到指定大小，確保了所有輸入圖像具有相同的尺寸，其中label使用的是最近鄰插值法，保持邊界的清晰度

  2. 隨機水平翻轉，隨即旋轉，增加數據多樣性，數據角度多樣性，以幫助模型學習位置不變性以及對不同方向的適應性

  3. 隨即應用高斯濾波，根據一定概率應用高斯濾波，模擬不同程度的圖像模糊，提高模型對不清晰圖像的魯棒性

  4. 將圖像（numpy數組）轉換爲pytorch張量，歸一化，使得像素值的範圍從[0,255]縮放到[0.0,1.0]

  5. 基於ImageNet數據集計算得出的均值和標準差作爲標準化參數對圖像進行標準化

  6. 將label轉換爲Pytorch的長整型張量（numpy 數組），label如果不是numpy數組，如果是列表或者PIL圖像

     **WHY NEED ToTensor?** 对于使用预训练模型时，pytorch的模型期望输入在 [0, 1] 范围内的张量格式，保證數據一致性

- 驗證和測試階段：去除了所有隨機過程，保留了Resize，歸一化和ToTensor

**ATTENTION!** 不管是什麽那個數據集，做完數據增强之後都加上了驗證形狀verify shapes，這樣就可以保證，data_loader得到的圖片尺寸相當

### Train

#### Deeplabv3 resnet-101

亮點：

- 使用CombinedLoss，結合了Focal Loss(1)和Dice Loss(0.5)，這樣就使得Focal Loss稍微偏重一點，Focal Loss主要解決類別不平衡的問題，適用於難分類的樣本；Dice Loss專注於提高分割的準確性，適用於處理小目標；事實上我們嘗試了不同的權重組合，但結果還是説明2:1的比重在驗證集上體現出相對較好的性能

- 采用OneCycleLR策略優化訓練過程：減少過擬合，高學習率階段，在達到最大學習率（0.01）時，模型參數更新幅度較大，有助於模型跳出局部最小值；使用餘弦退火策略，學習率在後70%的訓練時間內平滑地從高到低變化，有助於模型在高學習率階段探索更廣泛的參數空間後，逐漸精細化調整，提高泛化能力，最終學習率設置很低（0.00001），有助於模型在最優解附近進行微調，防止過擬合。

  **爲什麽要設置30% 70%?**  前30%的時間中，學習率快速上升到最大值，允許模型快速探索參數空間，加速收斂；后緩慢下降，在剩餘的70%時間裡，學習率從最大值（0.01）緩慢下降到最終值（0.00001）。

  ![image-20240715043548785](https://gcore.jsdelivr.net/gh/stellariumImpl/CDN/pic/image-20240715043548785.png)

- 使用Automatic Mixed Precision（AMP），自動化了精度訓練的過程，提高計算效率，節省内存

  ```python
  scaler = GradScaler()
  
  with torch.cuda.amp.autocast():
      outputs = model(images)
      loss = criterion(outputs, labels)
  
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

- 選擇三種評價指標mIOU，像素精確度，Dice係數，從以下問題的幾個方面考慮使用的優劣性

  |                    | mIOU | 像素精確度 | Dice係數   |
  | ------------------ | ---- | ---------- | ---------- |
  | 類別平衡敏感度     | 高   | 低         | 中         |
  | 小目標敏感度       | 中   | 低         | 高         |
  | 多分類問題適用性   | 高   | 中         | 低(二分類) |
  | 對錯誤類別的敏感度 | 高   | 低         | 中         |

  雖然是多分類問題，但每個類別的Dice係數仍然可以直觀解釋預測和真實分割的重叠程度，并且對小目標極爲敏感，對於不平衡（比如該數據集中有些圖片只有dirt和tree-trunk）的多酚類問題數據集中很有用，與mIOU和像素精確度互補

- 定期保存检查点：自动保存最佳模型和定期检查点

- 详细的日志记录：使用logging模块记录训练过程

- 模型本身，選用更深的網絡層數101，ResNet本身解決的就是深層網絡梯度消失問題，并且更深的網絡結構允許模型學習更複雜抽象的特徵，有助於捕捉圖像中的細節，更大的感受野有利於理解大尺度上下文信息，對語義分割任務相當重要，同時更深的網絡也具有更強的泛化能力。

| 特性         | ResNet101 | ResNet50 |
| ------------ | --------- | -------- |
| 深度         | 更深      | 較淺     |
| 參數量       | 更多      | 較少     |
| 計算成本     | 更高      | 較低     |
| 特徵提取能力 | 更強      | 較弱     |
| 感受野       | 更大      | 較小     |
| 精度（通常） | 更高      | 較低     |
| 訓練速度     | 較慢      | 較快     |
| 部署難度     | 較高      | 較低     |

- backbone

基礎模型為預訓練的Deeplabv3模型，backbone為Resnet-101，自定義了分類器，替換原有的最後一層。使用預訓練權重，有助於提高模型的初始性能，加速訓練。

​	自定義分類器header，原始的Deeplabv3分類器被替換成一個更複雜的結構：

​	a. 額外的卷積層，使用3*3捲積，padding=1保持特徵圖像的尺寸不變，增加了模型的容量，允許學習		複雜的特徵；

​	b. 批量歸一化，有助於穩定訓練過程，加速收斂

​	c. ReLU激活函數，非綫性，增强模型的表達能力

​	d. 使用BatchNorm，Dropout設置0.5丟棄一半的神經元，減少過擬合的風險

​	e. 最終分類層，使用1*1的捲積將特徵映射到所需的類別數量num_classes，這樣可以生成更精細的分		割結果

### Mask2former Swin L

對於模型本身，使用了预训练的 Mask2Former 模型（基于 Swin Transformer 的 COCO 全景分割版本），为模型提供了强大的特征提取能力和对复杂场景的理解能力。

backbone:

冻结了大部分预训练参数，这有助于保留预训练模型学到的通用特征。只解冻最后 5 层，允许这些层适应新的任务。

```python
# 冻结大部分参数
for param in self.mask2former.parameters():
    param.requires_grad = False
# 解冻最后几层
trainable_layers = list(self.mask2former.named_children())[-5:]
for name, layer in trainable_layers:
    for param in layer.parameters():
    	param.requires_grad = True
```

輸出層的適應：修改了`class_predictor`以匹配新的类别数

添加了一个额外的 3*3 捲積層：将 100 个查询通道转换为所需的类别数

前向传播过程：利用预训练模型生成初步的掩码查询，通过额外的卷积层处理这些查询，生成最终的类别预测。

```python
def forward(self, pixel_values):
    outputs = self.mask2former(pixel_values=pixel_values)
    masks = outputs.masks_queries_logits  # Shape: [batch_size, 100, height, width]
    masks = self.extra_conv(masks)  # Shape: [batch_size, num_classes, height, width]
    return masks
```

- Method collection 1:

  - 使用CombinedLoss，結合了Focal Loss(1)和Dice Loss(0.5)

  - 只优化需要梯度的参数 (`requires_grad=True`)，初始学习率设为5e-5，权重衰减(L2正则化)设为0.01,有助于防止过拟合

    *AdamW是Adam优化器的一个变体,它对权重衰减的处理更加合理,通常能获得更好的泛化性能。*

  - 学习率调度器，使用ReduceLROnPlateau调度器，监控验证损失('min'模式)，当验证损失在5个epoch内没有改善时,将学习率降低到原来的10%，有助于在训练后期微调模型,突破性能瓶颈
  - 數據增强：默認設置，與data_load部分的增强策略一致

- Method collection 2:
  - 數據增强，train階段新增了随机颜色抖动 (亮度、对比度、饱和度)，有些圖像光照强烈，随机颜色抖动可以增强模型對不同光照條件的適應能力，同時減少對特定顔色特徵的過度依賴，有助於模型關注物體的形狀和紋理特徵，not just color；随机裁剪，改善模型对局部特征的理解，迫使模型学习识别物体的局部特征，而不仅仅依赖于完整的物体形状
  - 使用 CosineAnnealingLR 学习率调度器，更平滑地使学习率下降，在Method collection 1中，學習率下降階段速度過快，在epoch 30就降低為0
  - 降低了梯度裁剪的阈值，稳定训练
  - 優化器設置，将优化器的 weight_decay 降低，lr=1e-4，减少正则化强度
  - 使用CombinedLoss，結合了Focal Loss(0.75)和Dice Loss(0.25)，减少对难分类样本的过度关注，稍微降低 Focal Loss的权重（1.0 →0.75）以防止模型过度关注极难分类的样本，可能是噪声或异常值；Focal Loss 有时可能导致训练不稳定，希望通過略微降低其权重來缓解这个问题。
  - Dice loss（0.5 → 0.25），由於Dice loss通常對大區域敏感，所以相对减少 Dice Loss 的影响有助于模型更好地处理小目标和精细结构，更多地关注像素级别的准确性
  - 降低損失函數整體的權重來降低总体损失值，Method collection 1中使用的是基于损失值的调度器`ReduceLROnPlateau`
 
<!-- ### 接下來要做的事情

在原有基礎上，添加Unet 和 FCN resnet 50 的敘述。

爲什麽在miou之外，我們要選擇了'pixel_acc': val_pixel_acc, 'dice': val_dice

###  -->


### 任務規範和數據集理解






​	

