### Plan
- Split the dataset
- Load data
- Select models (U-Net, DeepLab, FCN) for training, compare different models; adjust loss functions, choose optimizers, tune hyperparameters, calculate evaluation metrics (IoU, Dice coefficient)
- Apply test set to evaluate trained model performance, check generalization ability, verify for overfitting; visualize semantic segmentation on test set images; set up model saving and loading mechanisms
- Optimize dataset splitting, address gaps in the three sets
- Write paper, collect references

### Issues
- Color and trainId mapping seemed incorrect, why was the background all black? (Resolved)
- How to more intuitively record detailed logs during training and validation processes, how to use visualization tools like TensorBoard to observe changes in loss and mIoU? (Resolved)
- Current data volume is insufficient, only using V-01 image set (Resolved)

### Usage Instructions
Download the [dataset](https://doi.org/10.25919/5hzc-5p73) yourself and place it adjacent to the project root directory. Currently using V-01, V-02, and V-03.
All notebook file contents can be run normally to obtain results.
- data_split_optimizer.ipynb file splits the dataset
- group_work_deeplabv3_resnet101.ipynb includes training, prediction, and visualization for deeplabv3_resnet101
- group_work_fcn_resnet50.ipynb includes training, prediction, and visualization for fcn_resnet50
- group_work_mask2former.ipynb includes training, prediction, and visualization for mask2former
- group_work_unet.ipynb includes training, prediction, and visualization for unet

### Environment Configuration
```
conda create -n group_work python=3.9
conda activate group_work
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
# Some errors may occur, but they're not critical. If you need to resolve them:
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
# The environment already has torchvision 0.11.2 correctly installed, compatible with PyTorch 1.10.1
# Install torchaudio version compatible with PyTorch 1.10.1
pip install torchaudio==0.10.1 
# If you encounter any problems installing torchaudio
conda install torchaudio==0.10.1 -c pytorch
```
