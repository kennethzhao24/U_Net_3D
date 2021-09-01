import torch.nn as nn
from torch.nn import functional as F
import torch

class UNet3D(nn.Module):
    def __init__(self, f_maps=[32, 64, 128, 256], in_channels=1, out_channels=13, isTrain=True):
        super(UNet3D, self).__init__()

        self.isTrain = isTrain
        #create encoder path consisting of Encoder modules. Depth of the encoder is equal to len(f_maps)
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num)
            encoders.append(encoder)
        
        self.encoders = nn.ModuleList(encoders)

        #create decoder path consisting of the Decoder modules. The length of the decoder is equal to len(f_maps) - 1
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            
            decoder = Decoder(in_feature_num, out_feature_num)
            decoders.append(decoder)
        
        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)


    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        
        #remove last
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if not self.isTrain:
            x = F.sigmoid(x)

        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True):
        super(Encoder, self).__init__()

        if apply_pooling:
            self.pooling = nn.MaxPool3d(2)
        else:
            self.pooling = None
        
        self.basic_module = DoubleConv(in_channels, out_channels, encoder=True)
    
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.basic_module = DoubleConv(in_channels, out_channels, encoder=False)
    
    def forward(self, encoder_features, x):
        x = F.interpolate(x, encoder_features.size()[2:])
        x = torch.cat((encoder_features, x), dim=1)
        x = self.basic_module(x)
        return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder):
        super(DoubleConv, self).__init__()
        if encoder:
            #encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            #decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        
        self.add_module('SingleConv1', SingleConv(conv1_in_channels, conv1_out_channels))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels, conv2_out_channels))

class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.add_module('batchnorm', nn.BatchNorm3d(in_channels))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('leaky_relu', nn.LeakyReLU(0.1, inplace=True))
