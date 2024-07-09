### 计划（maybe）

- 划分数据集 
- 载入数据
- 选择模型（U-Net、DeepLab、FCN）训练循环，比较不同模型
- 定义损失函数,选择优化器
- 应用测试集评估模型性能，检查泛化能力，检验是否过拟合，调整超参数，评估指标（IoU、Dice coefficient
- 设置模型保存和加载机制
- 可视化
- 实例分割? 全景分割?

### 使用方法

`data_split`->`data_load.py` 均可单独执行

### 配置环境

```
conda create -n group_work python=3.9
conda activate image_processing
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
# 然后会出现一些报错 其实无所谓 如果需要解决
pip uninstall torchaudio -y
pip show torch torchvision
# #Name: torch
# Version: 1.10.1
# Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
# Home-page: https://pytorch.org/
# Author: PyTorch Team
# Author-email: packages@pytorch.org
# License: BSD-3
# Location: e:\languages\anaconda3\envs\image_processing\lib\site-packages
# Requires: typing-extensions
# Required-by: torchvision
# ---
# Name: torchvision
# Version: 0.11.2
# Summary: image and video datasets and models for torch deep learning
# Home-page: https://github.com/pytorch/vision
# Author: PyTorch Core Team
# Author-email: soumith@pytorch.org
# License: BSD
# Location: e:\languages\anaconda3\envs\image_processing\lib\site-packages
# Requires: numpy, pillow, torch
# Required-by:
# 环境中已经正确安装了与 PyTorch 1.10.1 兼容的 torchvision 0.11.2 版本
# 安装与 PyTorch 1.10.1 兼容的 torchaudio 版本
pip install torchaudio==0.10.1 
# 如果在安装 torchaudio 时遇到任何问题
conda install torchaudio==0.10.1 -c pytorch
