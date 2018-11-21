from core.config import cfg
import torch
import torch.nn as nn
import math
def EPE(input, target):
    """
    Endpoint-error
    :param input: 
    :param target: 
    :return: 
    """
    return (target - input).abs()

class MultiScaleLoss(nn.Module):
    def __init__(self, scales, downscale, weights=None, loss='MSE'):
        super(MultiScaleLoss, self).__init__()
        self.scales=scales
        self.downscale = downscale
        
        self.weights = torch.tensor([0.32,0.28,0.02,0.01,0.095,0.005])
        self.EPE=torch.zeros(1)
        self.EPE_=torch.zeros(1)
        assert (len(self.weights) == scales)
        if type(loss) is str:
            assert (loss in ['L1', 'MSE', 'SmoothL1'])

            if loss == 'L1':
                self.loss = nn.L1Loss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
        else:
            self.loss = loss
        self.multiScales = [nn.AvgPool2d(self.downscale * (2 ** i), self.downscale * (2 ** i)) for i in range(scales)]

    

    def forward(self, input, target):
        b, channel, h, w = input[0].size()
        loss=[]
        if type(input) is tuple:
            out = 0
            for i, input_ in enumerate(input):
                target_ = self.multiScales[i](target)
                target_ = target_.view(b,-1).contiguous()
                input_  = input_.view(b,-1).contiguous()
                self.EPE_ = torch.nn.functional.smooth_l1_loss(input_, target_,reduction='none')
                self.EPE_ = self.EPE_.mean().type(torch.cuda.FloatTensor)
                if i==0:
                    self.EPE=self.EPE_
                out += self.EPE_
                #out += self.weights[i] * self.EPE_ # Compare EPE_ with A Variable of the same size, filled with zeros)
            out=out/self.scales
        else:
            out = self.loss(input, self.multiScales[0](target))
        return out,self.EPE

def _disp_loss(self,pred, targe, sparse=True):
    b, channel, h, w = pred.size()
    pred=pred.view(b,-1).contiguous()
    targe=targe.view(b,-1).contiguous()
    EPE_map = nn.SmoothL1Loss(pred, targe)
    positive = (targe > 0).long()
    loss = EPE_map.mean()
    return loss