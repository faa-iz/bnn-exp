import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np
prune = 90
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        #return tensor.sign()
        return (tensor >= 0).type(tensor.type()) - (tensor < 0).type(tensor.type())
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class Binarizet(Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.tensor = tensor
        #if quant_mode == 'det':
        out =  tensor.sign()
        return out
        #else:
        #    return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)

    #'''
    @staticmethod
    
    def backward(ctx, grad_output):
        #print(ctx)
        tensor= ctx.tensor
        #grad_input = (1 - torch.pow(torch.tanh(tensor), 2))

        out = grad_output#*grad_input

        perc = np.percentile(out.abs().cpu().numpy(),prune)
        mask = out.abs()>=perc
        out = out*mask
        return out
    #'''

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

#import torch.nn._functions as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input=Binarizet.apply(input)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        bw=Binarizet.apply(self.weight)
        out = nn.functional.linear(input, bw)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input = Binarizet.apply(input)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        bw=Binarizet.apply(self.weight)
        #print(self.weight)

        out = nn.functional.conv2d(input, bw, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
