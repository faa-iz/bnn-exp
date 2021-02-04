import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import math

class LSQ(Function):
    @staticmethod
    def forward(self, value, step_size, nbits, signed):
        print('forward')
        self.save_for_backward(value, step_size)
        self.other = nbits, signed

        #set levels
        if signed:
            Qn = -2**(nbits-1)
            Qp = 2**(nbits-1) - 1
        else:
            Qn = 0
            Qp = 2**nbits - 1

        v_bar = (value/step_size).round().clamp(Qn, Qp)
        v_hat = v_bar*step_size
        return v_hat

    @staticmethod
    def backward(self, grad_output):
        print('backward')
        value, step_size = self.saved_tensors
        nbits, signed = self.other

        #set levels
        if signed:
            Qn = -2**(nbits-1)
            Qp = 2**(nbits-1) - 1
        else:
            Qn = 0
            Qp = 2**nbits - 1

        grad_scale = 1.0 / math.sqrt(value.numel() * Qp)

        lower = (value/step_size <= Qn).float()
        higher = (value/step_size >= Qp).float()
        middle = (1.0 - higher - lower)

        grad_step_size = lower*Qn + higher*Qp + middle*(-value/step_size + (value/step_size).round())

        return grad_output*middle, (grad_output*grad_step_size*grad_scale).sum().unsqueeze(dim=0), None, None


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

def quantizeLSQ(v, s, p, isActivation=False):
    if isActivation:
        Qn = 0
        Qp = 2**p - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)
    else: # is weight
        Qn = -2**(p-1)
        Qp = 2**(p-1) - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)

    #quantize
    s = grad_scale(s, gradScaleFactor)
    vbar = round_pass((v/s).clamp(Qn, Qp))
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

class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.nbits = kwargs['nbits']
        self.step_size= Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.nbits = kwargs_q['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=3):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits)


    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.init_state.fill_(1)

        #w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)
        w_q = LSQ.apply(self.weight, self.step_size, self.nbits, True)

        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=3):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.init_state.fill_(1)

        #w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)
        w_q = LSQ.apply(self.weight, self.step_size, self.nbits, True)

        return F.linear(x, w_q, self.bias)

class ActLSQ(_ActQ):
    def __init__(self, nbits=3):
        super(ActLSQ, self).__init__(nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        #x_q = quantizeLSQ(x, self.step_size, self.nbits, isActivation=True)
        x_q = LSQ.apply(x, self.step_size, self.nbits, False)

        return x_q

class PartialSumLSQ(_ActQ):
    def __init__(self, nbits=3):
        super(PartialSumLSQ, self).__init__(nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        #x_q = quantizeLSQ(x, self.step_size, self.nbits)
        x_q = LSQ.apply(x, self.step_size, self.nbits, True)

        return x_q
