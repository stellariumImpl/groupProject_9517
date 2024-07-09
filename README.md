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
