import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter


import numpy as np

nbits=1
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
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

    @staticmethod
    def backward(ctx, grad_output):
        #print(ctx)
        tensor= ctx.tensor
        grad_input = (torch.tanh(1/(100*tensor))).abs() * grad_output
        return grad_input, None, None


class LSQbi(Function):
    @staticmethod
    def forward(self, value, step_size, nbits):
        #print('forward2')
        #print('-------------')
        #print(step_size.data)
        #value  =  torch.where(value > step_size,step_size,value)
        #value  =  torch.where(value < -step_size,-step_size,value)

        self.save_for_backward(value, step_size)
        self.other = nbits

        #set levels
        Qn = -1
        Qp = 1

        #v_bar = (value >= 0).type(value.type()) - (value < 0).type(value.type()
        v_bar = value.sign()
        v_hat = v_bar*step_size
        return v_hat

    @staticmethod
    def backward(self, grad_output):
        #print('backward2')
        value, step_size = self.saved_tensors
        nbits = self.other

        #set levels
        Qn = -1
        Qp = 1
        grad_scale = 1.0 / (math.sqrt(value.numel() * Qp))

        lower = (value/step_size <= Qn-0.5).float()
        higher = (value/step_size >= Qp+0.5).float()
        middle = (1.0 - higher - lower)

        gradLower = 1#(Qn - (value/step_size)).clamp(0,1)
        gradHigher = -1#(Qp - (value/step_size)).clamp(-1,0)
        gradMiddle = value.sign()-(value/step_size)

        grad_weight = nn.functional.tanh(-(step_size*value.sign())+value).abs()

        weight_grad = (1 - torch.pow(torch.tanh(value), 2))

        #grad_step_size = -lower*Qn + higher*Qp + middle*(-value/step_size + (value/step_size).round())
        grad_step_size = lower*gradLower + higher*gradHigher + middle*gradMiddle

        return grad_output*-grad_step_size*grad_scale, (grad_output*grad_step_size*grad_scale).mean().unsqueeze(dim=0), None


class scale_out(Function):

    @staticmethod
    def forward(self, stride, padding, dilation, groups, out, weight, input, scale):
        #print('forward2')
        #print('-------------')
        #print(step_size.data)
        real_out = nn.functional.conv2d(input, weight, None, stride,padding, dilation, groups)
        self.save_for_backward(real_out, scale)


        out = out*scale
        return out

    @staticmethod
    def backward(self, grad_output):
        #print('backward2')
        real_out, scale = self.saved_tensors



        #set levels
        Qn = -1
        Qp = 1
        grad_scale = 1.0 / (math.sqrt(real_out.numel()))


        grad = scale - real_out.abs()
        grad =  grad.clamp(-0.25,0.25)
        grad = grad*grad_scale

        return None, None, None, None,None,None,None,(grad_output*grad).sum().unsqueeze(dim=0)



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




class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = Parameter(torch.ones(self.weight.size(0)))
        #self.alpha = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if self.init_state == 0:
            init1 = self.weight.abs().view(self.weight.size(0), -1).mean(-1)
            init1_ = self.weight.abs().mean()
            init2 =  input.abs().mean()
            self.alpha.data.copy_(torch.ones(self.weight.size(0)).cuda() * init1)
            '''
            if input.size(1) != 3:
                self.alpha.data.copy_(torch.ones(1).cuda() * init1_*init2)
            else:
                self.alpha.data.copy_(torch.ones(1).cuda() * init1_)
            '''
            #self.alpha.data.copy_(torch.ones(1).cuda() * init1_*init2)
            self.beta.data.copy_(torch.ones(1).cuda() * init2)
            #self.init_state.fill_(1)

        if input.size(1) != 3:
            #input_c = input.clamp(-1,1)
            inputq = Binarizet.apply(input)
            #inputq = LSQbi.apply(input,self.beta,1)
        else:
            inputq = input




        wq=Binarizet.apply(self.weight)
        #wq = LSQbi.apply(self.weight, self.alpha,1)

        out = nn.functional.conv2d(inputq, wq, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        out = out * self.alpha.view(1,out.shape[1],1,1) * self.beta

        #out =  scale_out.apply(self.stride,self.padding,self.dilation,self.groups, out,self.weight,input,self.alpha)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)


        return out
