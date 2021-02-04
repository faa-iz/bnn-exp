import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.nn.functional as tnnf
import numpy as np

def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeLSQ(v, s, p):
    #set levels
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1
    if p==1 or p==-1: #-1 is ternary
        Qn = -1
        Qp = 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel())
    else:
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)

    #quantize
    s = grad_scale(s, gradScaleFactor)
    vbar=round_pass((v/s).clamp(Qn, Qp))
    if p==1:
        vbar = Binarize(vbar)
    vhat = vbar*s
    return vhat

class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.nbits = kwargs['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class QuantizeConv2d(_Conv2dQ):
    def __init__(self, *kargs, **kwargs):
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        #Quantize Partial Sums
        if self.init_state == 0:
            self.step_size.data.copy_(2*out.abs().mean() / math.sqrt(2**(self.nbits-1) - 1))
            self.init_state.fill_(1)

        out = quantizeLSQ(out, self.step_size, self.nbits)

        return out

