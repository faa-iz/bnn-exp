import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter

import numpy as np

class Binarize(Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        #if quant_mode == 'det':
        out =  tensor.sign()
        return out
        #else:
        #    return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)

    @staticmethod
    def backward(ctx, grad_output):
        print(ctx)
        tensor= ctx.saved_tensors
        grad_input = (1 - torch.pow(torch.tanh(tensor), 2)) * grad_output
        return grad_input, None, None

'''
def Binarize(tensor,quant_mode='det'):
       if quant_mode == 'det':
            return tensor.sign()

        else:
            return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)
'''


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
            input.data=Binarize().apply(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize().apply(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = Parameter(torch.zeros(self.weight.size(0)))
        self.register_buffer('init_state', torch.zeros(1))


    def forward(self, input):


        if self.init_state == 0:
            init1 = self.weight.abs().view(self.weight.size(0), -1).mean(-1)
            init2 =  input.abs().mean()
            self.alpha.data.copy_(torch.ones(self.weight.size(0)).cuda() * init1*init2)
            self.init_state.fill_(1)


        if input.size(1) != 3:
            input.data = Binarize().apply(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        bw=Binarize().apply(self.weight)
        #print(self.weight.shape)



        #print(self.alpha.shape)
        #print(bw.shape)
        sw = bw
        if input.size(1) != 3:
            sw = bw*self.alpha.view(bw.size(0),1,1,1)


        out = nn.functional.conv2d(input, sw, None, self.stride,
                                   self.padding, self.dilation, self.groups)




        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
