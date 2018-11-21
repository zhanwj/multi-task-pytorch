import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
def costVolume(leftFeature,rightFeature,max_displacement):
    cost = torch.zeros(leftFeature.size()[0], leftFeature.size()[1]*2, max_displacement,  leftFeature.size()[2],  leftFeature.size()[3])
    for i in range(max_displacement):
        if i > 0 :
            cost[:, :leftFeature.size()[1], i, :,i:]   = leftFeature[:,:,:,i:]
            cost[:, leftFeature.size()[1]:, i, :,i:] = rightFeature[:,:,:,:-i]
        else:
            cost[:, :leftFeature.size()[1], i, :,:]   = leftFeature
            cost[:, leftFeature.size()[1]:, i, :,:]   = rightFeature
        cost = cost.contiguous()
    return cost

def costVolume2(leftFeature,rightFeature,max_displacement):
    cost = torch.zeros(leftFeature.size()[0], leftFeature.size()[1]*2, max_displacement,  leftFeature.size()[2],  leftFeature.size()[3])

    for b in range(cost.size()[0]):
        i=0
        while i < cost.size()[1]:
            print(i)
            for j in range(max_displacement):
                if j>0:
                    cost[b,i,j,:,j:]=leftFeature[b,i//2,:,j:]
                    cost[b,i+1,j,:,j:]=rightFeature[b,i//2,:,:-j]
                else:
                    cost[b,i,j,:,:]=leftFeature[b,i//2,...]
                    cost[b,i+1,j,:,:]=rightFeature[b,i//2,...]
            i+=2
    return cost


class PPMBilinear3D(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),channelsReduction=1024):
        super(PPMBilinear3D, self).__init__()
        self.use_softmax = use_softmax
        self.channelsReduction=channelsReduction
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        #self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.aspp_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        cost_channels = channelsReduction
        self.stack0   = self._createStack(cost_channels,cost_channels//2,stride2=1)
        self.stack1   = self._createStack(cost_channels//2,cost_channels//2,stride2=1)
        
        self.stack1_1 = self._createStack(cost_channels//2,cost_channels)
        self.stack1_2 = self._createStack(cost_channels,cost_channels)
        self.stack1_3 = self._Deconv3D(cost_channels,cost_channels)
        self.stack1_4 = self._Deconv3D(cost_channels,cost_channels//2)

        self.stack2_1 = self._createStack(cost_channels//2,cost_channels)
        self.stack2_2 = self._createStack(cost_channels,cost_channels)
        self.stack2_3 = self._Deconv3D(cost_channels,cost_channels)
        self.stack2_4 = self._Deconv3D(cost_channels,cost_channels//2)

        self.stack3_1 = self._createStack(cost_channels//2,cost_channels)
        self.stack3_2 = self._createStack(cost_channels,cost_channels)
        self.stack3_3 = self._Deconv3D(cost_channels,cost_channels)
        self.stack3_4 = self._Deconv3D(cost_channels,cost_channels//2)

        self.out1=self._output(cost_channels//2,out_planes=19)
        self.out2=self._output(cost_channels//2,out_planes=19)
        self.out3=self._output(cost_channels//2,out_planes=19)
        
        self.reduce = nn.Sequential(
            nn.Conv2d(512,self.channelsReduction,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(channelsReduction)
        )


        #self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        #self.dropout_deepsup = nn.Dropout2d(0.1)
    def _createStack(self,inplace=512,out_planes=512,kernel_size=3,stride1=1,stride2=2,bias=False,padding=1):
        return nn.Sequential(
            nn.Conv3d(inplace,out_planes,kernel_size=3,stride=stride1,groups=self.channelsReduction//2,padding=1,bias=False),
            nn.Conv3d(out_planes,out_planes,kernel_size=3,stride=stride2,groups=self.channelsReduction//2,padding=1,bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
            )
    def _Deconv3D(self,inplace,out_planes,kernel_size=3,stride=2,padding=1,out_padding=1,bias=False):
        return nn.ConvTranspose3d(inplace,out_planes,kernel_size,stride,padding,out_padding,bias=bias)

    def _output(self,inplace=256,out_planes=1,kernel_size=3,stride=1,padding=1,bias=False):
        return nn.Sequential(
            nn.Conv3d(inplace,inplace,kernel_size,padding,stride,bias=bias),
            nn.Conv3d(inplace,out_planes,kernel_size,padding,stride,bias=bias)
            )

    def forward(self, conv_out, segSize=None):
            
        conv5 = conv_out[-1]
        

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.aspp_last(ppm_out)
        
        left, right=torch.split(x, 1, dim=0)
        cost = costVolume(left,right,40)
        print("cost:",cost.shape)
        
        stack0=self.stack0(cost)
        print("stack0:",stack0.shape)
        
        stack1=self.stack1(stack0)
        print("stack1:",stack1.shape)
        
        stack1_1=self.stack1_1(stack1)
        print("stack1_1:",stack1_1.shape)
        stack1_2=self.stack1_2(stack1_1)
        print("stack1_2:",stack1_2.shape)
        stack1_3=self.stack1_3(stack1_2)+stack1_1 
        print("stack1_3:",stack1_3.shape)
        stack1_4=self.stack1_4(stack1_3)+stack0
        print("stack1_4:",stack1_4.shape)

        stack2_1=self.stack2_1(stack1_4)+stack1_3
        print("stack2_1:",stack2_1.shape)  
        stack2_2=self.stack2_2(stack2_1)
        print("stack2_2:",stack2_2.shape)
        stack2_3=self.stack2_3(stack2_2)+stack1_1
        print("stack2_3:",stack2_3.shape) 
        stack2_4=self.stack2_4(stack2_3)+stack0
        print("stack2_4:",stack2_4.shape)

        stack3_1=self.stack3_1(stack2_4)+stack2_3
        print("stack3_1",stack3_1.shape) 
        stack3_2=self.stack3_2(stack3_1)
        print("stack3_2",stack3_2.shape) 
        stack3_3=self.stack3_3(stack3_2)+stack1_1
        print("stack3_3",stack3_3.shape) 
        stack3_4=self.stack3_4(stack3_3)+stack0
        print("stack3_4",stack3_4.shape)
        # deep sup

        output1=self.out1(stack1_4)
        print("output1:",output1.shape)
        output2=self.out1(stack2_4)+output1
        print("output2:",output2.shape)
        output3=self.out1(stack3_4)+output2
        print("output3:",output3.shape)


       

        return output1,output2,output3



inputs=[]
x=torch.ones((2,2048,96,96))
inputs.append(x)

psm=PPMBilinear3D()
psm.to('cuda')
out=psm(inputs)