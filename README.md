# 3D U-Net with PyTorch-Lightning

This repository includes Pytorch-Lightning implementation of 3D U-Net on SHREC2020 (Classification in cryo-electron tomograms) dataset.

### Installation
```
git clone https://github.com/kennethzhao24/3d_unet_pytorch_lightning
```

### Usage
First download the [SHREC 2020 dataset](www2.projects.science.uu.nl/shrec/cryo-et/). Unzip it and put it in directory '3d_unet_pytorch_lightning/data/shrec2020'
```
cd 3d_unet_pytorch_lightning
pip install -r requirements.txt
```
Before training, first run
```
python dataset/norm.py
```
- This will generate normalized reconstruction files, reconstruction_norm.mrc, which are used for training.

To train, run
```
python train.py 
```
To open tensorboard to monitor your training progress, run
```
tensorboard --logdir tb_logs --port 6006 --bindall
```
A sample checkpoint is included in this repo, to test it out and see the results, run
```
python test.py
```
 - Segmentation masks for each class, output segmentation metrics for each class and mean values will be generated.
 - ChimeraX is recommended for 3D visualization.
 
To get detection results, run
```
python detection.py
python eval.py
```
- This will generate detailed detection results. 

### Results

Metrics|1bxn|1qvr | 1s3x | 1u6g | 2cg9 | 3cf3 | 3d2f | 3gl1 | 3h84 | 3qm1 | 4cr2 |4d8q|
---|---|---|---| --- |--- |--- |--- |--- |--- |--- |--- |--- |
Accuracy | 0.9946| 0.9820 | 0.9281 | 0.9738 | 0.9558 | 0.9799 | 0.9749 | 0.9576 | 0.9752 | 0.9439 | 0.9795 | 0.9935 |
Precision | 1.0 | 0.9151 | 0.5837 | 0.8273 | 0.7354 | 0.8640 | 0.9091 | 0.8725 | 0.8434 | 0.6977 | 0.8452 | 0.9747 |
Recall | 0.9388 | 0.8584 | 0.4936 | 0.8387 | 0.7193 | 0.9076 | 0.7477 | 0.5677 | 0.8750 | 0.6224 | 0.9221 | 0.9500 |
F1 score | 0.9684 | 0.8858 | 0.5349 | 0.8330 | 0.7273 | 0.8853 | 0.8205 | 0.6878 | 0.8589 | 0.6579 | 0.8820 | 0.9620 |


### Reference
- [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
- [SHREC 2020: Classification in cryo-electron tomograms](https://www.sciencedirect.com/science/article/pii/S0097849320301126)
