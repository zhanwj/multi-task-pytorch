import torch
import numpy as np
import torch
import os
import sys
import functools
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
import torchvision.models as M
from PIL import Image
import torchvision
from core.config import cfg


class Conv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)

    def forward(self, x):
        return F.leaky_relu(self.conv.forward(x), negative_slope=0.1, inplace=True)


class TConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1):
        super(TConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=False)

    def forward(self, x):
        return F.leaky_relu(self.conv.forward(x), negative_slope=0.1, inplace=True)


class CorrelationLayer1D(nn.Module):
    def __init__(self, max_disp=40, stride_2=1):
        super(CorrelationLayer1D, self).__init__()
        self.max_displacement = max_disp
        self.stride_2 = stride_2

    def forward(self, x_1, x_2):
        x_1 = x_1
        x_2 = F.pad(x_2, (self.max_displacement * 2, 0, 0, 0))
        return torch.cat([torch.sum(x_1 * x_2[:, :, :, _y:_y + x_1.size(3)], 1).unsqueeze(1) for _y in
                          range(0, self.max_displacement * 2 + 1, self.stride_2)], 1)


class CorrelationLayer2D(nn.Module):
    def __init__(self, max_disp=20, stride_1=1, stride_2=1):
        super(CorrelationLayer2D, self).__init__()
        self.max_displacement = max_disp
        self.stride_1 = stride_1
        self.stride_2 = stride_2

    def forward(self, x_1):
        x_1 = x_1
        x_2 = F.pad(x_1, [self.max_displacement] * 4)
        return torch.cat([torch.sum(x_1 * x_2[:, :, _x:_x + x_1.size(2), _y:_y + x_1.size(3)], 1).unsqueeze(1) for _x in
                          range(0, self.max_displacement * 2 + 1, self.stride_1) for _y in
                          range(0, self.max_displacement * 2 + 1, self.stride_2)], 1)


class DispFulNet(nn.Module):
    def __init__(self, ngf=64):
        super(DispFulNet, self).__init__()

        ################ down
        self.conv1 = Conv(3, ngf, kernel_size=7, stride=2, padding=3)
        self.conv2 = Conv(ngf, ngf * 2, kernel_size=5, stride=2, padding=2)

        self.corr=CorrelationLayer1D(max_disp=40,stride_2=1)
        self.conv_rdi=nn.Sequential(nn.Conv2d(ngf*2,ngf,kernel_size=1,stride=1,padding=0),
                                    nn.ReLU(inplace=True))


        #self.conv3 = Conv(ngf*2, ngf * 4, kernel_size=5, stride=2, padding=2)
        self.conv3 = Conv(145, ngf * 4, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = Conv(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = Conv(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1)

        ################ extract
        self.pr64 = nn.Conv2d(ngf * 16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr32 = nn.Conv2d(ngf * 8, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr16 = nn.Conv2d(ngf * 4, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr8 = nn.Conv2d(ngf * 2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr4 = nn.Conv2d(ngf * 1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr2 = nn.Conv2d(ngf // 2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr1 = nn.Conv2d(20, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.pr64_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr32_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr16_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr8_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr4_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr2_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr1_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        ################ up

        self.upconv6 = TConv(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.upconv5 = TConv(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.upconv4 = TConv(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.upconv3 = TConv(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1)
        self.upconv2 = TConv(ngf * 1, ngf // 2, kernel_size=4, stride=2, padding=1)
        self.upconv1 = TConv(ngf // 2, ngf // 4, kernel_size=4, stride=2, padding=1)

        ################ iconv
        self.iconv6 = nn.Conv2d(ngf * 16 + 1, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv5 = nn.Conv2d(769, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv4 = nn.Conv2d(385, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv3 = nn.Conv2d(193, ngf * 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv2 = nn.Conv2d(97, ngf // 2, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        
    

    def forward(self,left):
        right=left        # upsampling method for intermediate disparity not specified in the paper, so use bilinear as default
        conv1a = self.conv1(left)
        print("conv1a:",conv1a.shape)
        conv1b = self.conv1(right)
        conv2a = self.conv2(conv1a)
        conv2b = self.conv2(conv1b)
        print("conv1b:",conv1b.shape)
        print("conv2a:",conv2a.shape)
        print("conv2b:",conv2b.shape)

        corr = self.corr(conv2a,conv2b)
        conv_rdi = self.conv_rdi(conv2a)
        print("corr:",corr.shape)
        print("conv_rdi:",conv_rdi.shape)

        conv3 = self.conv3(torch.cat((corr, conv_rdi), dim=1))
        print("conv3:",conv3.shape)
        #conv3=self.conv3(torch.cat((conv2a,conv2b),dim=1))
        conv3_1 = self.conv3_1(conv3)
        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)
        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)
        print("conv4:",conv4_1.shape)
        print("conv5:",conv5_1.shape)
        print("conv5_1:",conv5_1.shape)
        print("conv6:",conv6.shape)
        print("conv6_1:",conv6_1.shape)

        pr_64 = self.pr64(conv6_1)
        upconv6 = self.upconv6(conv6_1)
        print("upconv6:",upconv6.shape)
        print("conv5_1:",conv5_1.shape)
        print("pr_64:",pr_64.shape)
        print("pr64_up",self.pr64_up(pr_64).shape)
        
        iconv6 = self.iconv6(torch.cat([upconv6, conv5_1, self.pr64_up(pr_64)], 1))

        pr_32 = self.pr32(iconv6)
        upconv5 = self.upconv5(iconv6)
        iconv5 = self.iconv5(torch.cat([upconv5, conv4_1, self.pr32_up(pr_32)], 1))
        print("upconv5:",upconv5.shape)
        print("conv4_1:",conv4_1.shape)
        print("pr_32:",pr_32.shape)
        print("pr32_up",self.pr32_up(pr_32).shape)        

        pr_16 = self.pr16(iconv5)
        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(torch.cat([upconv4, conv3_1, self.pr16_up(pr_16)], 1))
        print("upconv4:",upconv4.shape)
        print("conv3_1:",conv3_1.shape)
        print("pr_16:",pr_16.shape)
        print("pr16_up",self.pr16_up(pr_16).shape)

        pr_8 = self.pr8(iconv4)
        upconv3 = self.upconv3(iconv4)
        iconv3 = self.iconv3(torch.cat([upconv3, conv2a, self.pr8_up(pr_8)], 1))
        print("upconv3:",upconv3.shape)
        print("conv2a:",conv2a.shape)
        print("pr_8:",pr_8.shape)
        print("pr8_up",self.pr8_up(pr_8).shape)

        pr_4 = self.pr4(iconv3)
        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([upconv2, conv1a, self.pr4_up(pr_4)], 1))
        print("upconv2:",upconv2.shape)
        print("conv1a:",conv1a.shape)
        print("pr_4:",pr_4.shape)
        print("pr4_up",self.pr4_up(pr_4).shape)

        pr_2 = self.pr2(iconv2)
        upconv1 = self.upconv1(iconv2)
        print("pr2:",pr_2.shape)
        print("upconv1:",upconv1.shape)
        print("left:",left.shape)
        print("pr2_up:",self.pr2_up(pr_2).shape)

        pr_1 = self.pr1(torch.cat((upconv1, left, self.pr2_up(pr_2)),1))
        print("pr_1",pr_1.shape)

        return pr_64, pr_32, pr_16, pr_8, pr_4, pr_2, pr_1



################ for reference

class WarpOperate(nn.Module):

    def __init__(self,b, h,w,batchNorm=True):
        super(WarpOperate, self).__init__()
        H, W = h,w
        B=b
        xx = torch.arange(0, W).view(1,-1).repeat(H,1).cuda()
        yy = torch.arange(0, H).view(-1,1).repeat(1,W).cuda()
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        self.grid = Variable(grid, requires_grad=False).cuda()
        self.mask = Variable(torch.ones(
                      1, 3, H, W),requires_grad=True).cuda()
    
    def forward(self,x,flo):
        B, C, H, W = x.size()
        grid = self.grid
        vgrid = grid + flo
        
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = self.mask
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.999] = 0
        mask[mask>0] = 1
        
        return output*mask


class DispResNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(DispResNet, self).__init__()
        channels=64

        self.batchNorm = batchNorm
        self.conv1 = self._conv(self.batchNorm, 13, channels, kernel_size=5, stride=1)
        self.conv2 = self._conv(self.batchNorm, channels, 2*channels, kernel_size=5, stride=2)
        self.conv2_1=self._conv(self.batchNorm,2*channels,2*channels,stride=1)

        self.conv3 = self._conv(self.batchNorm, 2*channels, 4*channels,  stride=2)
        self.conv3_1 = self._conv(self.batchNorm, 4*channels, 4*channels,stride=1)

        self.conv4 = self._conv(self.batchNorm, 4*channels, 8*channels,stride=2)
        self.conv4_1 = self._conv(self.batchNorm, 8*channels, 8*channels,stride=1)

        self.conv5 = self._conv(self.batchNorm, 8*channels, 16*channels, stride=2)
        self.conv5_1 = self._conv(self.batchNorm, 16*channels, 16*channels,stride=1)
#########################################################################################
        self.res_16 = self._conv(self.batchNorm,16*channels,1,stride=1)
        self.res_8 = self._conv(self.batchNorm, 8*channels, 1,stride=1)
        self.res_4 = self._conv(self.batchNorm, 4*channels, 1,stride=1)
        self.res_2 = self._conv(self.batchNorm, 2*channels, 1,stride=1)
        self.res_1 = self._conv(self.batchNorm, 2*channels+1, 1,stride=1)
#########################################################################################
        self.iconv5 = nn.Conv2d(16*channels+1,8*channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.iconv4 = nn.Conv2d(16*channels+1, 8*channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv3 = nn.Conv2d(8*channels+1, channels * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv2 = nn.Conv2d(4*channels+1, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
  #################################################################################
        
    

        self.deconv4 = self._deconv(16*channels+1, 8*channels)
        self.deconv3 = self._deconv(8*channels+1, 4*channels)
        self.deconv2 = self._deconv(4*channels+1, 2*channels)
        self.deconv1 = self._deconv(2*channels+1, channels)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,
                                      0.02 / n)  # this modified initialization seems to work better, but it's very hacky
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    
    

    def _conv(self,batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                            bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                            bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
        
        
    def _predict_disp(self,in_planes):
        return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        
    def _deconv(self,in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def _downsampling(self,input,size=None,scale_factor=0.5,mode='bilinear',align_corners=False):
        return nn.functional.interpolate(input,size,scale_factor,mode,align_corners
        )    
       
    
    
                

        

    def forward(self, x):
        left=x[0]
        right=x[1]
        pr_s1=x[2]
        
        left_s=self._warp(left,pr_s1)
        err=torch.abs(left_s-left)

        conv1=self.conv1(torch.cat((left,right,left_s,err,pr_s1),dim=1))
        conv2=self.conv2(conv1)
        conv2_1=self.conv2_1(conv2)
        conv3=self.conv3(conv2_1)
        conv3_1=self.conv3_1(conv3)
        conv4=self.conv4(conv3_1)
        conv4_1=self.conv4_1(conv4)
        conv5=self.conv5(conv4_1)
        conv5_1=self.conv5_1(conv5)
        print("conv1:",conv1.shape)
        print("conv2:",conv2.shape)
        print("conv2_1:",conv2_1.shape)
        print("conv3:",conv3.shape)
        print("conv3_1:",conv3_1.shape)
        print("conv4:",conv4.shape)
        print("conv4_1:",conv4_1.shape)
        print("conv5:",conv5.shape)
        print("conv5_1:",conv5_1.shape)



        res_16=self.res_16(conv5_1)
        pr_s1_16=self._downsampling(pr_s1,scale_factor=0.0625)
        pr_s2_16=pr_s1_16+res_16
        print("res_16",res_16.shape)
        print("pr_s1_16",pr_s1_16.shape)
        print("pr_s2_16",pr_s2_16.shape)
        
       
        pr_s1_8=self._downsampling(pr_s1,scale_factor=0.125)
        upconv4=self.deconv4(torch.cat((conv5_1,pr_s2_16),dim=1))
        print("upconv4:",upconv4.shape)
        print("conv4_1:",conv4_1.shape)
        print("pr_s2_16",pr_s2_16.shape)
        iconv4=self.iconv4(torch.cat((upconv4,conv4_1,pr_s1_8),dim=1))
        res_8=self.res_8(iconv4)
        pr_s2_8=pr_s1_8+res_8
        
        pr_s1_4=self._downsampling(pr_s1,scale_factor=0.25)
        upconv3=self.deconv3(torch.cat((conv4_1,pr_s2_8),dim=1))
        iconv3=self.iconv3(torch.cat((upconv3,conv3_1,pr_s1_4),dim=1))
        res_4=self.res_4(iconv3)
        pr_s2_4=pr_s1_4+res_4
        
        pr_s1_2=self._downsampling(pr_s1,scale_factor=0.5)
        upconv2=self.deconv2(torch.cat((conv3_1,pr_s2_4),dim=1))
        iconv2=self.iconv2(torch.cat((upconv2,conv2_1,pr_s1_2),dim=1))
        res_2=self.res_2(iconv2)
        pr_s2_2=pr_s1_2+res_2


        upconv1=self.deconv1(torch.cat((iconv2,pr_s2_2),dim=1))
        res_1=self.res_1(torch.cat((upconv1,conv1,pr_s1),dim=1))
        pr_s2=pr_s1+res_1
        print("pr_s2:",pr_s2.shape)

        return pr_s2




class DispFulNetSubset(nn.Module):
    def __init__(self, ngf=128):
        super(DispFulNetSubset, self).__init__()

        ################ down
        
        self.conv4 = Conv(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = Conv(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = Conv(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1)

        ################ extract
        self.pr64 = nn.Conv2d(ngf * 16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr32 = nn.Conv2d(ngf * 8, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr16 = nn.Conv2d(ngf * 4, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr8 = nn.Conv2d(ngf * 2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr4 = nn.Conv2d(ngf * 2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr2 = nn.Conv2d(ngf // 2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.pr1 = nn.Conv2d(ngf // 4+1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.pr64_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr32_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr16_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr8_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr4_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr2_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.pr1_up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)


        self.upconv6 = TConv(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.upconv5 = TConv(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.upconv4 = TConv(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.upconv3 = TConv(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1)
        self.upconv2 = TConv(ngf * 2, ngf // 2, kernel_size=4, stride=2, padding=1)
        self.upconv1 = TConv(ngf // 2, ngf // 4, kernel_size=4, stride=2, padding=1)
  
        ################ iconv
        self.iconv6 = nn.Conv2d(ngf * 16 + 1, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv5 = nn.Conv2d(ngf * 8+ngf * 4+1, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv4 = nn.Conv2d(ngf * 4+ngf * 2+1, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv3 = nn.Conv2d(ngf*3+1, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv2 = nn.Conv2d(ngf+1, ngf // 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.re = nn.Conv2d(ngf*20+19,ngf*4,kernel_size=3,stride=1,padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,
                                      0.02 / n)  # this modified initialization seems to work better, but it's very hacky
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
         

    def forward(self,dispFeature,semseg,left,FeatureMap):
        
        conv1a, _ = torch.split(FeatureMap[0], cfg.TRAIN.IMS_PER_BATCH, dim=0)    #64channels
        #_ , conv1a = torch.split(conv1a, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        conv2a, _ = torch.split(FeatureMap[1], cfg.TRAIN.IMS_PER_BATCH, dim=0)   #128channels
        #_ , conv2a =  torch.split(conv2a, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        _, layer4 = torch.split(FeatureMap[4], cfg.TRAIN.IMS_PER_BATCH, dim=0)
        conv3_1=dispFeature  #
        
        conv3_1rdi =self.re(torch.cat((conv3_1,layer4,semseg),dim=1))
        conv4 = self.conv4(conv3_1rdi)
        conv4_1 = self.conv4_1(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)
        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)
       
        pr_64 = self.pr64(conv6_1)
        upconv6 = self.upconv6(conv6_1)
      
        
        iconv6 = self.iconv6(torch.cat([upconv6, conv5_1, self.pr64_up(pr_64)], 1))

        pr_32 = self.pr32(iconv6)
        upconv5 = self.upconv5(iconv6)

        iconv5 = self.iconv5(torch.cat([upconv5, conv4_1, self.pr32_up(pr_32)], 1))
      
        pr_16 = self.pr16(iconv5)
        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(torch.cat([upconv4, conv3_1rdi, self.pr16_up(pr_16)], 1))
     
        pr_8 = self.pr8(iconv4)
        upconv3 = self.upconv3(iconv4)
        iconv3 = self.iconv3(torch.cat([upconv3,conv2a,self.pr8_up(pr_8)], 1))
        

        pr_4 = self.pr4(iconv3)
        upconv2 = self.upconv2(iconv3)
        iconv2 =self.iconv2(torch.cat([upconv2, conv1a,self.pr4_up(pr_4)], 1))
        
        pr_2 = self.pr2(iconv2)
        upconv1 = self.upconv1(iconv2)

        pr_1 = self.pr1(torch.cat((upconv1, self.pr2_up(pr_2)),1))
        
        return  pr_1,pr_2,pr_4,pr_8,pr_16,pr_32


class DispResNetSubset(nn.Module):
    
    expansion = 1

    def __init__(self, batchNorm=True):
        super(DispResNetSubset, self).__init__()
        channels=64

        self.batchNorm = batchNorm
        self.conv1 = self._conv(self.batchNorm, 13, channels, kernel_size=5, stride=1)
        self.conv2 = self._conv(self.batchNorm, channels, 2*channels, kernel_size=5, stride=2)
        self.conv2_1=self._conv(self.batchNorm,2*channels,2*channels,stride=1)

        self.conv3 = self._conv(self.batchNorm, 2*channels, 4*channels,  stride=2)
        self.conv3_1 = self._conv(self.batchNorm, 4*channels, 4*channels,stride=1)

        self.conv4 = self._conv(self.batchNorm, 4*channels, 8*channels,stride=2)
        self.conv4_1 = self._conv(self.batchNorm, 8*channels, 8*channels,stride=1)

        self.conv5 = self._conv(self.batchNorm, 8*channels, 16*channels, stride=2)
        self.conv5_1 = self._conv(self.batchNorm, 16*channels, 16*channels,stride=1)

        self.res_16 = self._conv(self.batchNorm,16*channels,1,stride=1)
        self.res_8 = self._conv(self.batchNorm, 8*channels, 1,stride=1)
        self.res_4 = self._conv(self.batchNorm, 4*channels, 1,stride=1)
        self.res_2 = self._conv(self.batchNorm, 2*channels, 1,stride=1)
        self.res_1 = self._conv(self.batchNorm, 2*channels+1, 1,stride=1)

        self.iconv5 = nn.Conv2d(16*channels+1,8*channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.iconv4 = nn.Conv2d(16*channels+1, 8*channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv3 = nn.Conv2d(8*channels+1, channels * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.iconv2 = nn.Conv2d(4*channels+1, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        
    

        self.deconv4 = self._deconv(16*channels+1, 8*channels)
        self.deconv3 = self._deconv(8*channels+1, 4*channels)
        self.deconv2 = self._deconv(4*channels+1, 2*channels)
        self.deconv1 = self._deconv(2*channels+1, channels)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,
                                      0.02 / n)  # this modified initialization seems to work better, but it's very hacky
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _warp(self,x,flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        #vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.999] = 0
        mask[mask>0] = 1
        
        return output*mask
    
    

    def _conv(self,batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                            bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                            bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
        
        
    def _predict_disp(self,in_planes):
        return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        
    def _deconv(self,in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def _downsampling(self,input,size=None,scale_factor=0.5,mode='bilinear',align_corners=False):
        return nn.functional.interpolate(input,size,scale_factor,mode,align_corners
        )    
       
    
    
                

        

    def forward(self, x):
        left=x[0]
        right=x[1]
        pr_s1=x[2]
        
        left_s=self._warp(left,pr_s1)
        err=torch.abs(left_s-left)

        conv1=self.conv1(torch.cat((left,right,left_s,err,pr_s1),dim=1))
        conv2=self.conv2(conv1)
        conv2_1=self.conv2_1(conv2)
        conv3=self.conv3(conv2_1)
        conv3_1=self.conv3_1(conv3)
        conv4=self.conv4(conv3_1)
        conv4_1=self.conv4_1(conv4)
        conv5=self.conv5(conv4_1)
        conv5_1=self.conv5_1(conv5)
        print("conv1:",conv1.shape)
        print("conv2:",conv2.shape)
        print("conv2_1:",conv2_1.shape)
        print("conv3:",conv3.shape)
        print("conv3_1:",conv3_1.shape)
        print("conv4:",conv4.shape)
        print("conv4_1:",conv4_1.shape)
        print("conv5:",conv5.shape)
        print("conv5_1:",conv5_1.shape)



        res_16=self.res_16(conv5_1)
        pr_s1_16=self._downsampling(pr_s1,scale_factor=0.0625)
        pr_s2_16=pr_s1_16+res_16
        print("res_16",res_16.shape)
        print("pr_s1_16",pr_s1_16.shape)
        print("pr_s2_16",pr_s2_16.shape)
        
       
        pr_s1_8=self._downsampling(pr_s1,scale_factor=0.125)
        upconv4=self.deconv4(torch.cat((conv5_1,pr_s2_16),dim=1))
        print("upconv4:",upconv4.shape)
        print("conv4_1:",conv4_1.shape)
        print("pr_s2_16",pr_s2_16.shape)
        iconv4=self.iconv4(torch.cat((upconv4,conv4_1,pr_s1_8),dim=1))
        res_8=self.res_8(iconv4)
        pr_s2_8=pr_s1_8+res_8
        
        pr_s1_4=self._downsampling(pr_s1,scale_factor=0.25)
        upconv3=self.deconv3(torch.cat((conv4_1,pr_s2_8),dim=1))
        iconv3=self.iconv3(torch.cat((upconv3,conv3_1,pr_s1_4),dim=1))
        res_4=self.res_4(iconv3)
        pr_s2_4=pr_s1_4+res_4
        
        pr_s1_2=self._downsampling(pr_s1,scale_factor=0.5)
        upconv2=self.deconv2(torch.cat((conv3_1,pr_s2_4),dim=1))
        iconv2=self.iconv2(torch.cat((upconv2,conv2_1,pr_s1_2),dim=1))
        res_2=self.res_2(iconv2)
        pr_s2_2=pr_s1_2+res_2


        upconv1=self.deconv1(torch.cat((iconv2,pr_s2_2),dim=1))
        res_1=self.res_1(torch.cat((upconv1,conv1,pr_s1),dim=1))
        pr_s2=pr_s1+res_1
        print("pr_s2:",pr_s2.shape)

        return pr_s2
