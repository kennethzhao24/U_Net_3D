import argparse
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mrcfile
from unet import UNet3D
from utils.metrics import seg_metrics
import pytorch_lightning as pl
from dataset.preprocessing import SHREC_2020_Dataset
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from utils.misc import combine
import sys

parser = argparse.ArgumentParser(description='Testing for 3D U-Net models')

parser.add_argument('--dataset', help='dataset type', default='shrec')
parser.add_argument('--gpu_ids', default='0,1')
parser.add_argument('--f_maps', help='Convolution size', default=[32, 64, 128, 256])
parser.add_argument('--checkpoints', help='Checkpoint directory', default="./checkpoints/sample_checkpoint.ckpt")
parser.add_argument('--threshold', help='segmentation threshold', default=0.5)


args = parser.parse_args()

dataset = args.dataset
gpu_ids = args.gpu_ids
f_maps = args.f_maps
checkpoints = args.checkpoints
threshold = args.threshold


class UNetTest(pl.LightningModule):
    def __init__(self):
        super(UNetTest, self).__init__()
        self.model = UNet3D(f_maps=f_maps, isTrain=False)  
    def forward(self, x):
        return self.model(x)

# Check GPU Availability
if gpu_ids != '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
if torch.cuda.is_available():

    device = torch.device("cuda:0")

# load trained checkpoints to model
model = UNetTest.load_from_checkpoint(checkpoints)
model = model.to(device)
model.eval()

test_data = DataLoader(SHREC_2020_Dataset(mode='test'), shuffle=False, batch_size=1)

data_shape = [200, 512, 512]

precision_batch = []
recall_batch = []
iou_batch = []
f1_batch = []
seg_outputs = []

print("Start testing...")

for i, (img, label) in enumerate(test_data):
  img = img.to(device)
  label = label.float()
  label = label.view(-1, 13, 72, 72, 72)
  label = label.to(device)
  outputs = model(img)
  
  precision, recall, iou, f1 = seg_metrics(outputs, label, isTrain=False, threshold=threshold)

  outputs = outputs.cpu().detach().numpy().squeeze()

  seg_outputs.append(outputs)

  precision_batch.append(precision)
  recall_batch.append(recall)
  iou_batch.append(iou)
  f1_batch.append(f1)

print("Testing complete!")

# convert list to numpy array
precision_batch = np.array(precision_batch)
recall_batch = np.array(recall_batch)
iou_batch = np.array(iou_batch)
f1_batch = np.array(f1_batch)

print("Generating segmentation metrics and prediction mrc file(s)...")

# make table and output segmentation result per class
classes = ['0', '3cf3', '1s3x', '1u6g', '4cr2', '1qvr', '3h84', '2cg9', '3qm1', '3gl1', '3d2f', '4d8q', '1bxn']

seg_outputs = np.array(seg_outputs)
seg_outputs = np.transpose(seg_outputs, (1,0,2,3,4))
t = PrettyTable(['Class','Precision', 'Reall', 'IoU', 'F1'])
for i in range(13):
  t.add_row([classes[i],
             precision_batch.mean(axis=0)[i],
             recall_batch.mean(axis=0)[i],
             iou_batch.mean(axis=0)[i],
             f1_batch.mean(axis=0)[i]
             ]
            )
  mrc_name = './result/shrec_mask/shrec_result_class_{}.mrc'.format(classes[i])
  union_result = mrcfile.new(mrc_name, overwrite=True)
  union_data = combine(seg_outputs[i], shape=data_shape)
  union_result.set_data(union_data)
  union_result.close()
  print("{}/{}".format(i+1, 13))

print('======= Segmentation metrics (threshold = %.1f)=======' % threshold) 
print(t)

# ouput mean metrics
print("""
Mean Precision: %.4f
Mean Recall: %.4f
Mean IoU: %.4f
Mean F1: %.4f
"""
%(np.mean(precision_batch), 
  np.mean(recall_batch), 
  np.mean(iou_batch), 
  np.mean(f1_batch)
)
)