print("spn not implement!!!!!! please uncomment!!")
import torch.nn as nn
from core.config import  cfg
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_,constant_
from modeling.pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
from lib.nn import SynchronizedBatchNorm2d

class SPN(nn.Module):
#docstring for SPN
    def __init__(self):
        super(SPN, self).__init__()
        self.connection_ways=3
        self.left_to_right=GateRecurrent2dnoind(True,False)
        self.right_to_left=GateRecurrent2dnoind(True,True)
        self.bottom_to_up =GateRecurrent2dnoind(False,True)
        self.up_to_bottom =GateRecurrent2dnoind(False,False)
        self.guide_conv1=nn.Sequential(
                nn.Conv2d(cfg.MODEL.NUM_CLASSES, cfg.SPN.DIM, kernel_size=4,padding=1,stride=2,bias=False))
        self.guide_conv2=nn.Sequential(
                nn.Conv2d(256,cfg.SPN.DIM*12,kernel_size=3,padding=1,stride=1,bias=False))
        self.elt_resize_deconv=nn.Sequential(                      #1/2
            nn.Conv2d(cfg.SPN.DIM, cfg.SPN.DIM*2, 3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.SPN.DIM*2, cfg.MODEL.NUM_CLASSES, kernel_size=3, padding=1, stride=1, bias=False)
            )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _gNorm(self,glist):
        return_list=[]
        g_norm_list=[]
        sum_abs=glist[0].abs()
        for i in range(1,len(glist)):
            sum_abs+=glist[i].abs()
        mask_need_norm=sum_abs.ge(1)
        mask_need_norm=mask_need_norm.float()

        for i in range(len(glist)):
            g_norm_list.append(torch.div(glist[i],sum_abs))
            return_list.append(torch.add(-mask_need_norm,1)*glist[i]+mask_need_norm*g_norm_list[i])
        return return_list

    def forward(self,featureMap,guidance):
        featureMap=self.guide_conv1(featureMap)
        guidance=self.guide_conv2(guidance)
        gate=[]
        gate=torch.split(guidance,split_size_or_sections=cfg.SPN.DIM,dim=1)
        
        G_left_to_right=gate[:self.connection_ways]
        G_right_to_left=gate[self.connection_ways:2*self.connection_ways]
        G_bottom_to_up=gate[2*self.connection_ways:4*self.connection_ways]
        G_up_to_bottom=gate[-self.connection_ways:]
        G_left_to_right=self._gNorm(G_left_to_right)
        G_right_to_left=self._gNorm(G_right_to_left)
        G_bottom_to_up =self._gNorm(G_bottom_to_up)
        G_up_to_bottom =self._gNorm(G_up_to_bottom)
        
        output_left_to_right=self.left_to_right(featureMap,G_left_to_right[0],G_left_to_right[1],G_left_to_right[2])
        output_right_to_left=self.right_to_left(featureMap,G_right_to_left[0],G_right_to_left[1],G_right_to_left[2])
        output_bottom_to_up =self.bottom_to_up(featureMap,G_bottom_to_up[0], G_bottom_to_up[1], G_bottom_to_up[2])
        output_up_to_bottom =self.up_to_bottom(featureMap,G_up_to_bottom[0], G_up_to_bottom[1], G_up_to_bottom[2])
        output_max=torch.max(torch.max(torch.max(output_left_to_right,output_right_to_left),output_up_to_bottom),output_bottom_to_up)
        
        if cfg.SPN.SPN_ITERS>1:
            for i in range(cfg.SPN.SPN_ITERS-1):
                output_left_to_right=self.left_to_right(output_max,G_left_to_right[0],G_left_to_right[1],G_left_to_right[2])
                output_right_to_left=self.right_to_left(output_max,G_right_to_left[0],G_right_to_left[1],G_right_to_left[2])
                output_bottom_to_up =self.bottom_to_up (output_max,G_bottom_to_up[0], G_bottom_to_up[1], G_bottom_to_up[2])
                output_up_to_bottom =self.up_to_bottom (output_max,G_up_to_bottom[0], G_up_to_bottom[1], G_up_to_bottom[2])
                output_max=torch.max(torch.max(torch.max(output_left_to_right,output_right_to_left),output_up_to_bottom),output_bottom_to_up)
        #return output_max
        output_max = nn.functional.interpolate(output_max,scale_factor=2,mode='bilinear', align_corners=False)
        predict1x=self.elt_resize_deconv(output_max)
        assert predict1x.shape[2] == cfg.SEM.INPUT_SIZE[0] and predict1x.shape[3] == cfg.SEM.INPUT_SIZE[1], 'spn output of size %s' % str(predict1x.shape)
        return predict1x


