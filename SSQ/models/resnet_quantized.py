import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import numpy as np
from .quantized_modules import  QuantizeConv2d
from .lsq import Conv2dLSQ, LinearLSQ, ActLSQ, PartialSumLSQ
import matplotlib.pyplot as plt
import scipy.stats as stats

__all__ = ['resnet_quantized']

##################################  PARTIAL SUMS PARAMETERS CONTROL #################################
num_model = 3
num_bit = 4


############################################################################

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=100)

def Quantizeconv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2dLSQ(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, nbits=num_model)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, QuantizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def split_tensor_128(xp):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)

    return x1,x2,x3,x4,x5,x6,x7,x8,x9

def split_tesnsor_256(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92

def split_tesnsor_384(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x13 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x23 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x33 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x43 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x53 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x63 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x73 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x83 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x93 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93

def split_tesnsor_512(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x13 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x23 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x33 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x43 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x53 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x63 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x73 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x83 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x93 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x14 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x24 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x34 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x44 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x54 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x64 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x74 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x84 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x94 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bntan=True):
        super(BasicBlock, self).__init__()

        ###########################################     CONV1       ##########################################
        max_size = 128
        groups = math.ceil(inplanes / max_size)

        #padding outside the conv
        self.padding1 = nn.ZeroPad2d(1)
        input_dem = inplanes
        inplanes = min(input_dem,max_size)

        self.act1 = ActLSQ()
        self.conv1 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq1 = PartialSumLSQ(nbits=num_bit)

        self.act2 = ActLSQ()
        self.conv2 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq2 = PartialSumLSQ(nbits=num_bit)

        self.act3 = ActLSQ()
        self.conv3 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq3 = PartialSumLSQ(nbits=num_bit)

        self.act4 = ActLSQ()
        self.conv4 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq4 = PartialSumLSQ(nbits=num_bit)

        self.act5 = ActLSQ()
        self.conv5 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq5 = PartialSumLSQ(nbits=num_bit)

        self.act6 = ActLSQ()
        self.conv6 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq6 = PartialSumLSQ(nbits=num_bit)

        self.act7 = ActLSQ()
        self.conv7 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq7 = PartialSumLSQ(nbits=num_bit)

        self.act8 = ActLSQ()
        self.conv8 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq8 = PartialSumLSQ(nbits=num_bit)

        self.act9 = ActLSQ()
        self.conv9 = Quantizeconv1x1(inplanes, planes, stride)
        self.plsq9 = PartialSumLSQ(nbits=num_bit)


        if (input_dem > 128):     #Input channels = 256
            self.act12 = ActLSQ()
            self.conv12 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq12 = PartialSumLSQ(nbits=num_bit)

            self.act22 = ActLSQ()
            self.conv22 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq22 = PartialSumLSQ(nbits=num_bit)

            self.act32 = ActLSQ()
            self.conv32 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq32 = PartialSumLSQ(nbits=num_bit)

            self.act42 = ActLSQ()
            self.conv42 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq42 = PartialSumLSQ(nbits=num_bit)

            self.act52 = ActLSQ()
            self.conv52 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq52 = PartialSumLSQ(nbits=num_bit)

            self.act62 = ActLSQ()
            self.conv62 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq62 = PartialSumLSQ(nbits=num_bit)

            self.act72 = ActLSQ()
            self.conv72 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq72 = PartialSumLSQ(nbits=num_bit)

            self.act82 = ActLSQ()
            self.conv82 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq82 = PartialSumLSQ(nbits=num_bit)

            self.act92 = ActLSQ()
            self.conv92 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq92 = PartialSumLSQ(nbits=num_bit)

        if (input_dem > 256):       #Input channels = 384
            self.act13 = ActLSQ()
            self.conv13 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq13 = PartialSumLSQ(nbits=num_bit)

            self.act23 = ActLSQ()
            self.conv23 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq23 = PartialSumLSQ(nbits=num_bit)

            self.act33 = ActLSQ()
            self.conv33 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq33 = PartialSumLSQ(nbits=num_bit)

            self.act43 = ActLSQ()
            self.conv43 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq43 = PartialSumLSQ(nbits=num_bit)

            self.act53 = ActLSQ()
            self.conv53 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq53 = PartialSumLSQ(nbits=num_bit)

            self.act63 = ActLSQ()
            self.conv63 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq63 = PartialSumLSQ(nbits=num_bit)

            self.act73 = ActLSQ()
            self.conv73 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq73 = PartialSumLSQ(nbits=num_bit)

            self.act83 = ActLSQ()
            self.conv83 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq83 = PartialSumLSQ(nbits=num_bit)

            self.act93 = ActLSQ()
            self.conv93 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq93 = PartialSumLSQ(nbits=num_bit)

        if (input_dem > 384):       #Input channels = 512
            self.act14 = ActLSQ()
            self.conv14 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq14 = PartialSumLSQ(nbits=num_bit)

            self.act24 = ActLSQ()
            self.conv24 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq24 = PartialSumLSQ(nbits=num_bit)

            self.act34 = ActLSQ()
            self.conv34 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq34 = PartialSumLSQ(nbits=num_bit)

            self.act44 = ActLSQ()
            self.conv44 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq44 = PartialSumLSQ(nbits=num_bit)

            self.act54 = ActLSQ()
            self.conv54 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq54 = PartialSumLSQ(nbits=num_bit)

            self.act64 = ActLSQ()
            self.conv64 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq64 = PartialSumLSQ(nbits=num_bit)

            self.act74 = ActLSQ()
            self.conv74 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq74 = PartialSumLSQ(nbits=num_bit)

            self.act84 = ActLSQ()
            self.conv84 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq84 = PartialSumLSQ(nbits=num_bit)

            self.act94 = ActLSQ()
            self.conv94 = Quantizeconv1x1(inplanes, planes, stride)
            self.plsq94 = PartialSumLSQ(nbits=num_bit)


        ###########################################     END     ##########################################

        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)



        ###########################################     CONV2       ##########################################

        # max_size = 128
        inplanes = min(max_size,planes)
        groups = math.ceil(planes / max_size)
        self.padding2 = nn.ZeroPad2d(1)

        self.act1_2 = ActLSQ()
        self.conv1_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq1_2 = PartialSumLSQ(nbits=num_bit)

        self.act2_2 = ActLSQ()
        self.conv2_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq2_2 = PartialSumLSQ(nbits=num_bit)

        self.act3_2 = ActLSQ()
        self.conv3_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq3_2 = PartialSumLSQ(nbits=num_bit)

        self.act4_2 = ActLSQ()
        self.conv4_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq4_2 = PartialSumLSQ(nbits=num_bit)

        self.act5_2 = ActLSQ()
        self.conv5_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq5_2 = PartialSumLSQ(nbits=num_bit)

        self.act6_2 = ActLSQ()
        self.conv6_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq6_2 = PartialSumLSQ(nbits=num_bit)

        self.act7_2 = ActLSQ()
        self.conv7_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq7_2 = PartialSumLSQ(nbits=num_bit)

        self.act8_2 = ActLSQ()
        self.conv8_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq8_2 = PartialSumLSQ(nbits=num_bit)

        self.act9_2 = ActLSQ()
        self.conv9_2 = Quantizeconv1x1(inplanes, planes)
        self.plsq9_2 = PartialSumLSQ(nbits=num_bit)


        if(planes>128):
            self.act12_2 = ActLSQ()
            self.conv12_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq12_2 = PartialSumLSQ(nbits=num_bit)

            self.act22_2 = ActLSQ()
            self.conv22_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq22_2 = PartialSumLSQ(nbits=num_bit)

            self.act32_2 = ActLSQ()
            self.conv32_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq32_2 = PartialSumLSQ(nbits=num_bit)

            self.act42_2 = ActLSQ()
            self.conv42_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq42_2 = PartialSumLSQ(nbits=num_bit)

            self.act52_2 = ActLSQ()
            self.conv52_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq52_2 = PartialSumLSQ(nbits=num_bit)

            self.act62_2 = ActLSQ()
            self.conv62_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq62_2 = PartialSumLSQ(nbits=num_bit)

            self.act72_2 = ActLSQ()
            self.conv72_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq72_2 = PartialSumLSQ(nbits=num_bit)

            self.act82_2 = ActLSQ()
            self.conv82_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq82_2 = PartialSumLSQ(nbits=num_bit)

            self.act92_2 = ActLSQ()
            self.conv92_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq92_2 = PartialSumLSQ(nbits=num_bit)

        if (planes > 256):
            self.act13_2 = ActLSQ()
            self.conv13_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq13_2 = PartialSumLSQ(nbits=num_bit)

            self.act23_2 = ActLSQ()
            self.conv23_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq23_2 = PartialSumLSQ(nbits=num_bit)

            self.act33_2 = ActLSQ()
            self.conv33_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq33_2 = PartialSumLSQ(nbits=num_bit)

            self.act43_2 = ActLSQ()
            self.conv43_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq43_2 = PartialSumLSQ(nbits=num_bit)

            self.act53_2 = ActLSQ()
            self.conv53_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq53_2 = PartialSumLSQ(nbits=num_bit)

            self.act63_2 = ActLSQ()
            self.conv63_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq63_2 = PartialSumLSQ(nbits=num_bit)

            self.act73_2 = ActLSQ()
            self.conv73_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq73_2 = PartialSumLSQ(nbits=num_bit)

            self.act83_2 = ActLSQ()
            self.conv83_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq83_2 = PartialSumLSQ(nbits=num_bit)

            self.act93_2 = ActLSQ()
            self.conv93_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq93_2 = PartialSumLSQ(nbits=num_bit)

        if (planes > 384):
            self.act14_2 = ActLSQ()
            self.conv14_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq14_2 = PartialSumLSQ(nbits=num_bit)

            self.act24_2 = ActLSQ()
            self.conv24_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq24_2 = PartialSumLSQ(nbits=num_bit)

            self.act34_2 = ActLSQ()
            self.conv34_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq34_2 = PartialSumLSQ(nbits=num_bit)

            self.act44_2 = ActLSQ()
            self.conv44_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq44_2 = PartialSumLSQ(nbits=num_bit)

            self.act54_2 = ActLSQ()
            self.conv54_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq54_2 = PartialSumLSQ(nbits=num_bit)

            self.act64_2 = ActLSQ()
            self.conv64_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq64_2 = PartialSumLSQ(nbits=num_bit)

            self.act74_2 = ActLSQ()
            self.conv74_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq74_2 = PartialSumLSQ(nbits=num_bit)

            self.act84_2 = ActLSQ()
            self.conv84_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq84_2 = PartialSumLSQ(nbits=num_bit)

            self.act94_2 = ActLSQ()
            self.conv94_2 = Quantizeconv1x1(inplanes, planes)
            self.plsq94_2 = PartialSumLSQ(nbits=num_bit)


        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, x):
        #global partial_sums

        partial_sums = []
        inplanes = x.shape[1]
        max_size = 128
        groups = math.ceil(inplanes / max_size)
        #print(groups)

        residual = x.clone()
        xp = x
        xp = self.padding1(xp)

        #splitting x
        if(xp.shape[1]<=128):
            x1,x2,x3,x4,x5,x6,x7,x8,x9 = split_tensor_128(xp)
        elif(xp.shape[1]==256):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92= split_tesnsor_256(xp)
        elif (xp.shape[1] == 384):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92,x13,x23,x33,x43,x53,x63,x73,x83,x93 = split_tesnsor_384(xp)
        elif (xp.shape[1] == 512):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92,x13,x23,x33,x43,x53,x63,x73,x83,x93,x14,x24,x34,x44,x54,x64,x74,x84,x94 = split_tesnsor_512(xp)

        else:
            print(xp.shape)
            print("============ILLEGAL INPUT======================")

        out = []

        #Append quantized partial sums
        out.append(self.plsq1(self.conv1(self.act1(x1))))
        out.append(self.plsq2(self.conv2(self.act2(x2))))
        out.append(self.plsq3(self.conv3(self.act3(x3))))
        out.append(self.plsq4(self.conv4(self.act4(x4))))
        out.append(self.plsq5(self.conv5(self.act5(x5))))
        out.append(self.plsq6(self.conv6(self.act6(x6))))
        out.append(self.plsq7(self.conv7(self.act7(x7))))
        out.append(self.plsq8(self.conv8(self.act8(x8))))
        out.append(self.plsq9(self.conv9(self.act9(x9))))

        if (xp.shape[1]>128):

            out.append(self.plsq12(self.conv12(self.act12(x12))))
            out.append(self.plsq22(self.conv22(self.act22(x22))))
            out.append(self.plsq32(self.conv32(self.act32(x32))))
            out.append(self.plsq42(self.conv42(self.act42(x42))))
            out.append(self.plsq52(self.conv52(self.act52(x52))))
            out.append(self.plsq62(self.conv62(self.act62(x62))))
            out.append(self.plsq72(self.conv72(self.act72(x72))))
            out.append(self.plsq82(self.conv82(self.act82(x82))))
            out.append(self.plsq92(self.conv92(self.act92(x92))))

        if (xp.shape[1] > 256):
            out.append(self.plsq13(self.conv13(self.act13(x13))))
            out.append(self.plsq23(self.conv23(self.act23(x23))))
            out.append(self.plsq33(self.conv33(self.act33(x33))))
            out.append(self.plsq43(self.conv43(self.act43(x43))))
            out.append(self.plsq53(self.conv53(self.act53(x53))))
            out.append(self.plsq63(self.conv63(self.act63(x63))))
            out.append(self.plsq73(self.conv73(self.act73(x73))))
            out.append(self.plsq83(self.conv83(self.act83(x83))))
            out.append(self.plsq93(self.conv93(self.act93(x93))))

        if (xp.shape[1] > 384):
            out.append(self.plsq14(self.conv14(self.act14(x14))))
            out.append(self.plsq24(self.conv24(self.act24(x24))))
            out.append(self.plsq34(self.conv34(self.act34(x34))))
            out.append(self.plsq44(self.conv44(self.act44(x44))))
            out.append(self.plsq54(self.conv54(self.act54(x54))))
            out.append(self.plsq64(self.conv64(self.act64(x64))))
            out.append(self.plsq74(self.conv74(self.act74(x74))))
            out.append(self.plsq84(self.conv84(self.act84(x84))))
            out.append(self.plsq94(self.conv94(self.act94(x94))))

        output = torch.zeros(out[0].shape).cuda()

        for out_tensor in out:
            output = output + out_tensor

        output = self.bn1(output)
        xn = self.tanh1(output)

        xn = self.padding2(xn)
        inplanes = x.shape[1]
        groups = math.ceil(inplanes / max_size)

        if (xn.shape[1] <= 128):
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = split_tensor_128(xn)
        elif (xn.shape[1] == 256):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92 = split_tesnsor_256(xn)
        elif (xn.shape[1] == 384):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93 = split_tesnsor_384(
                xn)
        elif (xn.shape[1] == 512):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94 = split_tesnsor_512(
                xn)

        else:
            print(xn.shape)
            print("============ILLEGAL INPUT======================")

        out = []

        out.append(self.plsq1_2(self.conv1_2(self.act1_2(x1))))
        out.append(self.plsq2_2(self.conv2_2(self.act2_2(x2))))
        out.append(self.plsq3_2(self.conv3_2(self.act3_2(x3))))
        out.append(self.plsq4_2(self.conv4_2(self.act4_2(x4))))
        out.append(self.plsq5_2(self.conv5_2(self.act5_2(x5))))
        out.append(self.plsq6_2(self.conv6_2(self.act6_2(x6))))
        out.append(self.plsq7_2(self.conv7_2(self.act7_2(x7))))
        out.append(self.plsq8_2(self.conv8_2(self.act8_2(x8))))
        out.append(self.plsq9_2(self.conv9_2(self.act9_2(x9))))

        if(xn.shape[1]>128):
            out.append(self.plsq12_2(self.conv12_2(self.act12_2(x12))))
            out.append(self.plsq22_2(self.conv22_2(self.act22_2(x22))))
            out.append(self.plsq32_2(self.conv32_2(self.act32_2(x32))))
            out.append(self.plsq42_2(self.conv42_2(self.act42_2(x42))))
            out.append(self.plsq52_2(self.conv52_2(self.act52_2(x52))))
            out.append(self.plsq62_2(self.conv62_2(self.act62_2(x62))))
            out.append(self.plsq72_2(self.conv72_2(self.act72_2(x72))))
            out.append(self.plsq82_2(self.conv82_2(self.act82_2(x82))))
            out.append(self.plsq92_2(self.conv92_2(self.act92_2(x92))))

        if (xn.shape[1] > 256):
            out.append(self.plsq13_2(self.conv13_2(self.act13_2(x13))))
            out.append(self.plsq23_2(self.conv23_2(self.act23_2(x23))))
            out.append(self.plsq33_2(self.conv33_2(self.act33_2(x33))))
            out.append(self.plsq43_2(self.conv43_2(self.act43_2(x43))))
            out.append(self.plsq53_2(self.conv53_2(self.act53_2(x53))))
            out.append(self.plsq63_2(self.conv63_2(self.act63_2(x63))))
            out.append(self.plsq73_2(self.conv73_2(self.act73_2(x73))))
            out.append(self.plsq83_2(self.conv83_2(self.act83_2(x83))))
            out.append(self.plsq93_2(self.conv93_2(self.act93_2(x93))))

        if (xn.shape[1] > 384):
            out.append(self.plsq14_2(self.conv14_2(self.act14_2(x14))))
            out.append(self.plsq24_2(self.conv24_2(self.act24_2(x24))))
            out.append(self.plsq34_2(self.conv34_2(self.act34_2(x34))))
            out.append(self.plsq44_2(self.conv44_2(self.act44_2(x44))))
            out.append(self.plsq54_2(self.conv54_2(self.act54_2(x54))))
            out.append(self.plsq64_2(self.conv64_2(self.act64_2(x64))))
            out.append(self.plsq74_2(self.conv74_2(self.act74_2(x74))))
            out.append(self.plsq84_2(self.conv84_2(self.act84_2(x84))))
            out.append(self.plsq94_2(self.conv94_2(self.act94_2(x94))))



        output = torch.zeros(out[0].shape).cuda()

        for out_tensor in out:
            output = output + out_tensor


        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            #print("TRUEEEEEEEEEEEEEEEEEEEEE")
            residual = self.downsample(residual)


        #print(str(output.shape) + " ==>> " + str(residual.shape))
        output += residual
        #if self.do_bntan:
        output = self.bn2(output)
        output = self.tanh2(output)


        return output

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1,do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dLSQ(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes,do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        #p_s1,p_s2,p_s3,ps_4 = []
        x = self.act1(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.act2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x

class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 4
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)

        self.act1 = ActLSQ(nbits=8)
        self.conv1 = Conv2dLSQ(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, nbits=8)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*5, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 96*self.inflate, n, stride=2, do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(96*self.inflate)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()
        self.act2 = ActLSQ(nbits=8)
        self.fc = LinearLSQ(96*self.inflate, num_classes, nbits=8)

        init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }


def resnet_quantized(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])


    if dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    else:                                       #if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 18
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        '''
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])
        '''
