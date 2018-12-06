from core.config import  cfg
import torch
import torch.nn as nn
from torch.autograd import Variable
from modeling.pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
from modeling.cspn import Affinity_Propagate
from torch.nn.init import kaiming_normal_,constant_
class SPN(nn.Module):
    """docstring for SPN"""
    def __init__(self):
        super(SPN, self).__init__()
        self.connection_ways=cfg.SEM.SPN_CONNECTION_WAYS
        self.cspn = Affinity_Propagate()
        self.out1 = nn.Conv2d(cfg.MODEL.NUM_CLASSES, cfg.SEM.SPN_DIM, 3,  stride=1, padding=1)
        self.out2 = nn.Sequential(
                nn.Conv2d(cfg.SEM.SPN_DIM, cfg.SEM.SPN_DIM*2, 3,  stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cfg.SEM.SPN_DIM*2, cfg.MODEL.NUM_CLASSES, 3, stride=1, padding=1))
        #self.out2 = nn.Conv2d(cfg.SEM.SPN_DIM, cfg.MODEL.NUM_CLASSES, 3, groups=cfg.SEM.SPN_DIM, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    constant_(m.bias,0)
                
    def forward(self,featureMap,guidance):
        featureMap=self.out1(featureMap)
        #featureMap=nn.functional.interpolate(featureMap,scale_factor=1.0/2,mode='bilinear',align_corners=False)
        b, c, h, w = guidance.shape
        gate = guidance.view(b, 8, -1, h, w)
        sparse_depth = (nn.functional.softmax(featureMap,dim=1) > 0.8).float() * featureMap
        output_max = self.cspn(gate, featureMap,sparse_depth=sparse_depth)
        output_max = self.out2(output_max)
        #output_max = nn.functional.interpolate(output_max,scale_factor=2,mode='bilinear',align_corners=False)
        return output_max
