################
#
# Deep Flow Prediction for Cavity Flow - Neural Network Model
#
# U-Net architecture generator network for flow field prediction
#
# Author: Tesbo
#
################

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    """
    Weight initialization function
    Use normal distribution initialization for convolutional layers,
    and special initialization for batch normalization layers
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    """
    Basic building block of U-Net
    Parameters:
        in_c - Input channels
        out_c - Output channels
        name - Layer name
        transposed - Whether it's an upsampling layer (decoder)
        bn - Whether to use batch normalization
        relu - Whether to use ReLU (False uses LeakyReLU)
        size - Kernel size
        pad - Padding size
        dropout - Dropout probability
    """
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    
    if not transposed:
        # Encoder: downsampling
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        # Decoder: upsampling
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear'))
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block

# Generator network (U-Net architecture)
class TurbNetG(nn.Module):
    """
    Cavity flow prediction generator network
    Input: 3 channels (lid velocity X, lid velocity Y, Reynolds number)
    Output: 3 channels (pressure, velocity X, velocity Y)
    """
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        # Encoder (downsampling path)
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels*2, 'layer2', transposed=False, bn=True, relu=False, dropout=dropout)
        self.layer2b = blockUNet(channels*2, channels*2, 'layer2b', transposed=False, bn=True, relu=False, dropout=dropout)
        self.layer3 = blockUNet(channels*2, channels*4, 'layer3', transposed=False, bn=True, relu=False, dropout=dropout)
        self.layer4 = blockUNet(channels*4, channels*8, 'layer4', transposed=False, bn=True, relu=False, dropout=dropout, size=4)
        self.layer5 = blockUNet(channels*8, channels*8, 'layer5', transposed=False, bn=True, relu=False, dropout=dropout, size=2, pad=0)
        self.layer6 = blockUNet(channels*8, channels*8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, size=2, pad=0)
     
        # Decoder (upsampling path)
        self.dlayer6 = blockUNet(channels*8, channels*8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, size=2, pad=0)
        self.dlayer5 = blockUNet(channels*16, channels*8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels*16, channels*4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer3 = blockUNet(channels*8, channels*2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2b = blockUNet(channels*4, channels*2, 'dlayer2b', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2 = blockUNet(channels*4, channels, 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        """Forward pass with skip connections"""
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b = self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

# Discriminator (for adversarial training, optional)
class TurbNetD(nn.Module):
    """
    Discriminator network (used for adversarial training)
    """
    def __init__(self, in_channels1, in_channels2, ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv2d(in_channels1 + in_channels2, ch, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(ch, ch*2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(ch*4, ch*8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2d(ch*8, 1, 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2d(ch*2)
        self.bnc2 = nn.BatchNorm2d(ch*4)
        self.bnc3 = nn.BatchNorm2d(ch*8)

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2), 1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h)
        return h
