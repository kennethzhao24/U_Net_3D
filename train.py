import torch
from torch import optim
import torch.nn as nn
from utils.loss import DiceLoss, TverskyLoss
from utils.metrics import seg_metrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers
from dataset.preprocessing import SHREC_2020_Dataset
from model.unet import UNet3D
import argparse

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='Training for 3D U-Net models')
# dataset parameters
parser.add_argument('--block_size', help='block size', default=72)
# model parameters
parser.add_argument('--f_maps', help='feature size', default=[32, 64, 128, 256])

# training parameters
parser.add_argument('--loss_func', help='loss function type', default='Dice')
parser.add_argument('--learning_rate', type=int, default=5e-5)
parser.add_argument('--batch_size', help='batch size', default=32)
parser.add_argument('--max_epoch', help='number of epochs', default=100)


args = parser.parse_args()

block_size = args.block_size

f_maps = args.f_maps

loss_func = args.loss_func
lr = args.learning_rate
batch_size = args.batch_size
max_epoch = args.max_epoch

hparams = {'block_size': block_size,
           'f_maps': f_maps,
           'loss function': loss_func,
           'learning_rate': lr,
           'batch_size': batch_size,
           'max_epoch': max_epoch
           }

class UNetExperiment(pl.LightningModule):
    def __init__(self, hparams):
        super(UNetExperiment, self).__init__()

        self.hparams = hparams

        self.model = UNet3D(f_maps=self.hparams['f_maps'], out_channels=self.hparams['num_class']+1)

        if self.hparams['loss function'] == 'Dice':
            self.loss_function = DiceLoss()
        if self.hparams['loss function'] == 'Tversky':
            self.loss_function = TverskyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        img, label = train_batch
        output = self.forward(img)
        label = label.view(-1, self.hparams['num_class']+1,
                               self.hparams['block_size'],
                               self.hparams['block_size'],
                               self.hparams['block_size'],)
        loss = self.loss_function(output, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        img, label = val_batch
        output = self.forward(img)
        label = label.view(-1, self.hparams['num_class']+1,
                               self.hparams['block_size'],
                               self.hparams['block_size'],
                               self.hparams['block_size'],)
        loss = self.loss_function(output, label)
        precision, recall, f1_score, iou = seg_metrics(output, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True)
        self.log('val_f1', f1_score, on_step=False, on_epoch=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True)

    def train_dataloader(self):
        return DataLoader(SHREC_2020_Dataset(mode='train', 
                                             block_size=self.hparams['block_size'],
                                             num_class=self.hparams['num_class']),
                                             batch_size=self.hparams['batch_size'], 
                                             num_workers=32, 
                                             shuffle=True, 
                                             pin_memory=True)

    def val_dataloader(self):
        return DataLoader(SHREC_2020_Dataset(mode='val', 
                                             block_size=self.hparams['block_size'],
                                             num_class=self.hparams['num_class']),
                                             batch_size=self.hparams['batch_size'], 
                                             num_workers=32, 
                                             shuffle=True, 
                                             pin_memory=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                              lr=self.hparams['learning_rate'],
                              betas=(0.9, 0.99)
                              )
        return [optimizer]

checkpoint_callback = ModelCheckpoint(save_top_k=10, 
                                      monitor='val_loss',
                                      mode='min')

logger_name = "{}_multiclass_epoch_{}".format(hparams['loss function'], str(hparams['max_epoch']))

tb_logger = loggers.TensorBoardLogger("tb_logs", name=logger_name)

lr_monitor = LearningRateMonitor(logging_interval='step')

model = UNetExperiment(hparams=hparams)

runner = Trainer(max_epochs=hparams['max_epoch'],
                 logger=tb_logger,
                 gpus=[0,1], 
                 checkpoint_callback=checkpoint_callback,
                 callbacks=[lr_monitor],
                 distributed_backend='ddp',
                 precision=16,
                 profiler=True
                 )


runner.fit(model)
