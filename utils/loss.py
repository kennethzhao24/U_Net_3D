import torch
from torch import nn as nn
from torch.nn import functional as F
from .misc import flatten

# Dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = flatten(inputs)
        targets = flatten(targets)
        
        intersection = (inputs * targets).sum(-1)                            
        dice = (2.*intersection + smooth)/(inputs.sum(-1) + targets.sum(-1) + smooth)
        
        return 1 - dice.mean()

# Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.7, beta=0.3):

        if inputs.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            inputs = torch.cat((inputs, 1 - inputs), dim=0)
            targets = torch.cat((targets, 1 - targets), dim=0)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = flatten(inputs)
        targets = flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(-1)    
        FP = ((1-targets) * inputs).sum(-1)
        FN = (targets * (1-inputs)).sum(-1)
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky.mean()

# IoU Loss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = flatten(inputs)
        targets = flatten(targets)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum(-1)
        total = (inputs + targets).sum(-1)
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU.mean()

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = flatten(inputs)
        targets = flatten(targets)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

# Focal/Tversky Loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = flatten(inputs)
        targets = flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(-1)    
        FP = ((1-targets) * inputs).sum(-1)
        FN = (targets * (1-inputs)).sum(-1)
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky.mean())**gamma
                       
        return FocalTversky
