import torch
import torch.nn as nn
import torchvision
from lib.modeling import resnet as resnet
from lib.modeling.dispSeg_heads import ModelBuilder as ModelBuilder_dispSeg_heads
from lib.modeling.semseg_heads import ModelBuilder 
from lib.modeling.model_builder_3DSD import Generalized_3DSD
from torch.nn.init import kaiming_normal_,constant_
from core.config import  cfg
import torch.nn.functional as F


import time
timer=time.time
def costVolume2(leftFeature,rightFeature,max_displacement):
    cost = torch.zeros(leftFeature.size()[0], leftFeature.size()[1]*2, max_displacement,  leftFeature.size()[2],  leftFeature.size()[3]).to('cuda')

    for b in range(cost.size()[0]):
        i=0
        while i < cost.size()[1]:
            for j in range(max_displacement):
                if j>0:
                    cost[b,i,j,:,j:]=leftFeature[b,i//2,:,j:]
                    cost[b,i+1,j,:,j:]=rightFeature[b,i//2,:,:-j]
                else:
                    cost[b,i,j,:,:]=leftFeature[b,i//2,...]
                    cost[b,i+1,j,:,:]=rightFeature[b,i//2,...]
            i+=2
    return cost

def load_ckpt(model, ckpt):
    """Load checkpoint"""
    state_dict = {}
    for name in ckpt:
        if mapping[name]:
            state_dict[name] = ckpt[name]
    model.load_state_dict(state_dict, strict=False)



class disparityregression(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()
        self.SmoothL1Loss = nn.SmoothL1Loss(reduction='none') #elementwise_mean
    def forward(self, pred, pred_semseg, label_info, label_i=0, sparse=True, shape=(30,30)):
        targe = label_info['disp_label_0']
        bs, num_class,  h,w = pred.shape
        if sparse:
            targe = F.adaptive_max_pool2d(targe, shape)
        #calucate disp output
        #pred_scans = label_info['semseg_scans'].repeat(bs//cfg.TRAIN.IMS_PER_BATCH, 1, h, w)
        #pred_scans = (pred_semseg == pred_scans.long()).float()
        #pred_scans = pred_scans.unsqueeze(2)
        #pred_scans.requires_grad = False
        #pred_disp = pred*pred_scans
        #pred_disp = torch.sum(pred_disp, 1)
        disp_scans = label_info['disp_scans'].repeat(bs//cfg.TRAIN.IMS_PER_BATCH, 1, 1, 1)
        pred_semseg, _ = torch.max(pred_semseg, dim=1)
        pred_semseg = pred_semseg.unsqueeze(1)
        pred = pred + pred_semseg
        pred = F.softmax(pred, dim=1)
        disp_pred = torch.sum(pred*disp_scans, dim=1)
        #loss and pred
        EPE_map = self.SmoothL1Loss(disp_pred, targe)
        epe_pred = torch.abs(disp_pred - targe)
        #ignore false disp values
        positive = targe.ge(0)
        EPE_map = torch.masked_select(EPE_map, positive)
        epe_pred = torch.masked_select(epe_pred, positive)
        #normlization
        loss = EPE_map.mean()
        epe_pred = epe_pred.mean()
        return loss, epe_pred


class PSPNET(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PSPNET, self).__init__()
        self.builder=ModelBuilder()
        self.encoder = self.builder.build_encoder(
                arch=cfg.SEM.ARCH_ENCODER,
                fc_dim=2048,
                weights=''
                )
        self.decoder = self.builder.build_decoder(
                arch=cfg.SEM.DECODER_TYPE,
                fc_dim=2048,
                num_class=19,
                use_softmax=not self.training,
                weights='')


        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                constant_(m.weight,1)
                constant_(m.bias,0)

    def forward(self,data):
        x=self.decoder(self.encoder(data,return_feature_maps=True))
        return x

class GlassGCN(nn.Module):
    def __init__(self):
        super(GlassGCN,self).__init__()
        self.channelsReduction=cfg.SEM.SD_DIM
        self.ppm = []
        self.width=96
        self.height=96
        self.semseg=19
        self.max_displacement=48
        
        
        cost_channels = self.channelsReduction*2
        self.stack0   = self._createStack(cost_channels,cost_channels,stride1=1)
        
        self.stack1_1 = self._createStack(cost_channels,cost_channels*2)
        self.stack1_2 = self._createStack(cost_channels*2,cost_channels*4)
        self.stack1_3 = self._createStack(cost_channels*4,cost_channels*8)

        self.stack2_1 = self._Deconv3D(cost_channels*8,cost_channels*4)
        self.stack2_2 = self._Deconv3D(cost_channels*4,cost_channels*2)
        self.stack2_3 = self._Deconv3D(cost_channels*2,cost_channels)
        
        self.gcn1=GCNASPP(cost_channels*4,self.semseg,self.max_displacement//4,self.height//4,self.width//4,scale=2,pool_scales=(4,8,13,24))
        self.gcn2=GCNASPP(cost_channels*2,self.semseg,self.max_displacement//2,self.height//2,self.width//2,scale=1,pool_scales=(2,4,6,12))
        self.gcn3=GCNASPP(cost_channels,self.semseg,self.max_displacement,self.height,self.width,scale=0,pool_scales=(2,3,4,6))
        
        
        self.reduce = nn.Sequential(
            nn.Conv2d(512,self.channelsReduction,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(self.channelsReduction)
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
                constant_(m.weight,1)
                constant_(m.bias,0)

            #self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
            #self.dropout_deepsup = nn.Dropout2d(0.1)
    def _createStack(self,inplanes=512,planes=256,kernel_size=3,stride1=2,groups=19,stride2=1,bias=False,padding=1):
        return nn.Sequential(
            nn.Conv3d(inplanes,planes,kernel_size=3,stride=stride1,groups=cfg.GROUP_NORM.NUM_GROUPS,padding=1,bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes,planes,kernel_size=3,stride=stride2,groups=cfg.GROUP_NORM.NUM_GROUPS,padding=1,bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
            )
    def _Deconv3D(self,inplanes,planes,kernel_size=3,stride=2,padding=1,out_padding=1,groups=19,bias=False):
            return nn.ConvTranspose3d(inplanes,planes,kernel_size,stride,padding,out_padding,groups=cfg.GROUP_NORM.NUM_GROUPS,bias=bias)


    def forward(self, conv_out):
        x = self.reduce(conv_out)
        left, right=torch.split(x, 1, dim=0)
        cost = costVolume2(left,right,self.max_displacement)
        stack0=self.stack0(cost)
        stack1_1=self.stack1_1(stack0)
        stack1_2=self.stack1_2(stack1_1)
        stack1_3=self.stack1_3(stack1_2)
    
        stack2_1=self.stack2_1(stack1_3)
        stack2_2=self.stack2_2(stack2_1)
        stack2_3=self.stack2_3(stack2_2)
    
        if self.training:
            #gcn1=self.gcn1(stack2_1)
            #gcn2=self.gcn2(stack2_2)
            gcn3=self.gcn3(stack2_3)
            return gcn3
        else:
            gcn3=self.gcn3(stack2_3)
            return gcn3


class OfficialPSMNET(nn.Module):
    def __init__(self):
        super(OfficialPSMNET,self).__init__()
        self.channelsReduction=cfg.SEM.SD_DIM
        self.ppm = []
        self.semseg=19
        self.max_displacement=cfg.DISP.FEATURE_MAX_DISPLACEMENT
        
        #self.reduce = nn.Sequential(
        #    nn.Conv2d(512,self.channelsReduction,kernel_size=1,stride=1,bias=False),
        #)

        cost_channels = self.channelsReduction*2
        
        
        self.dres0 = nn.Sequential(convbn_3d(cost_channels, cost_channels//2, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(cost_channels//2, cost_channels//2, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(cost_channels//2, cost_channels//2, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(cost_channels//2, cost_channels//2, 3, 1, 1)) 

        self.dres2 = hourglass(self.channelsReduction)
        self.dres3 = hourglass(self.channelsReduction)
        self.dres4 = hourglass(self.channelsReduction)

        self.classif1 = nn.Sequential(convbn_3d(self.channelsReduction, self.channelsReduction, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.channelsReduction, self.semseg, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(self.channelsReduction, self.channelsReduction, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.channelsReduction, self.semseg, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(self.channelsReduction, self.channelsReduction, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.channelsReduction, self.semseg, kernel_size=3, padding=1, stride=1,bias=False))
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
                constant_(m.weight,1)
                constant_(m.bias,0)
    

    def forward(self, conv_out):
        x = conv_out
        left, right=torch.split(x, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        cost = costVolume2(left,right,self.max_displacement)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0


        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        return [cost3]



class PSMNET(nn.Module):
    def __init__(self):
        super(PSMNET,self).__init__()
        self.channelsReduction=cfg.SEM.SD_DIM
        self.ppm = []
        self.width=96
        self.height=96
        self.semseg=19
        self.max_displacement=cfg.DISP.FEATURE_MAX_DISPLACEMENT
        
        
        cost_channels = self.channelsReduction*2
        self.stack0   = self._createStack(cost_channels,cost_channels,stride1=1)
        self.stack1   = self._createStack(cost_channels,cost_channels,stride1=1)
        
        self.stack1_1 = self._createStack(cost_channels,cost_channels*2)
        self.stack1_2 = self._createStack(cost_channels*2,cost_channels*4)
        self.stack1_3 = self._Deconv3D(cost_channels*4,cost_channels*2)
        self.stack1_4 = self._Deconv3D(cost_channels*2,cost_channels)

        self.stack2_1 = self._createStack(cost_channels,cost_channels*2)
        self.stack2_2 = self._createStack(cost_channels*2,cost_channels*4)
        self.stack2_3 = self._Deconv3D(cost_channels*4,cost_channels*2)
        self.stack2_4 = self._Deconv3D(cost_channels*2,cost_channels)

        self.stack3_1 = self._createStack(cost_channels,cost_channels*2)
        self.stack3_2 = self._createStack(cost_channels*2,cost_channels*4)
        self.stack3_3 = self._Deconv3D(cost_channels*4,cost_channels*2)
        self.stack3_4 = self._Deconv3D(cost_channels*2,cost_channels)
        
        self.output = self._createStack(cost_channels,self.semseg,groups=1,stride1=1,stride2=1)
        """
        self.reduce = nn.Sequential(
            nn.Conv2d(512,self.channelsReduction,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(self.channelsReduction)
        )
        """
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
                constant_(m.weight,1)
                constant_(m.bias,0)

            #self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
            #self.dropout_deepsup = nn.Dropout2d(0.1)
    def _createStack(self,inplanes=512,planes=256,kernel_size=3,stride1=2,stride2=1,groups=cfg.GROUP_NORM.NUM_GROUPS,bias=False,padding=1):
        return nn.Sequential(
            nn.Conv3d(inplanes,planes,kernel_size=3,stride=stride1,groups=groups,padding=1,bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes,planes,kernel_size=3,stride=stride2,groups=groups,padding=1,bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
            )
    def _Deconv3D(self,inplanes,planes,kernel_size=3,stride=2,padding=1,out_padding=1,groups=19,bias=False):
            return nn.ConvTranspose3d(inplanes,planes,kernel_size,stride,padding,out_padding,groups=cfg.GROUP_NORM.NUM_GROUPS,bias=bias)


    def forward(self, conv_out):
        left, right=torch.split(x, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        cost = costVolume2(left,right,self.max_displacement)

        stack0=self.stack0(cost)
        stack1=self.stack1(stack0)

        stack1_1=self.stack1_1(stack1)
        stack1_2=self.stack1_2(stack1_1)
        stack1_3=self.stack1_3(stack1_2)+stack1_1
        stack1_4=self.stack1_4(stack1_3)+stack1
    
        stack2_1=self.stack2_1(stack1_4)+stack1_3
        stack2_2=self.stack2_2(stack2_1)
        stack2_3=self.stack2_3(stack2_2)+stack1_1
        stack2_4=self.stack2_4(stack2_3)+stack1

        stack3_1=self.stack3_1(stack2_4)+stack2_3
        stack3_2=self.stack3_2(stack3_1)
        stack3_3=self.stack3_3(stack3_2)+stack1_1
        stack3_4=self.stack3_4(stack3_3)+stack1

        out=self.output(stack3_4)

        return out


class DispSeg(nn.Module):
    def __init__(self):
        super(DispSeg,self).__init__()
        self.pspnet=PSPNET()
        self.glassGCN=OfficialPSMNET()

        self.last =nn.Conv2d(cfg.MODEL.NUM_CLASSES*2,cfg.SEM.SD_DIM,3,1,1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                constant_(m.weight,1)
                constant_(m.bias,0)

        if cfg.SEM.PSPNET_PRETRAINED_WEIGHTS is not None:
            print("loading pspnet weights")
            state_dict={}
            pretrained=torch.load(cfg.SEM.PSPNET_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            pretrained = pretrained['model']
            self.pspnet.load_state_dict(pretrained,strict=False)
            print("weights load success")
        
        if not cfg.SEM.PSPNET_REQUIRES_GRAD:
            for p in self.pspnet.parameters():
                p.requires_grad=False

        if not cfg.DISP.DISPSEG_REQUIRES_GRAD:
            for p in self.glassGCN.parameters():
                p.requires_grad=False
        #define 3D segdisp loss
        self.disparityregression = disparityregression()
        self.crit = nn.NLLLoss(ignore_index=255)
        self.crit_3d = nn.NLLLoss(ignore_index=255*cfg.DISP.MAX_DISPLACEMENT+1)
        self.conv_last_disp = nn.Conv3d(19, 1, 5, padding=2)
        self.conv_last_disp.apply(self._init_weights_kaiming)
        self.conv_last_segdisp = nn.Conv3d(19, 19, 3, padding=1)
        self.conv_last_segdisp.apply(self._init_weights_kaiming)
        self.conv_last_semseg = nn.Conv3d(19, 19, (cfg.DISP.MAX_DISPLACEMENT,1,1), padding=0)
        self.conv_last_semseg.apply(self._init_weights_kaiming)

    def _init_weights_kaiming(self,m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_uniform_(m.weight)
        if type(m) == nn.ConvTranspose2d:
            nn.init.kaiming_uniform_(m.weight)

    def pixel_acc(self, preds, label):
        #label = label.view(-1).contiguous()
        preds = torch.argmax(preds, dim=1)
        valid = (label != 255).long()
        acc_sum = torch.sum(valid*(preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def segdisp_loss(self, pred, label_info, label_i):
        bs, num_class, maxdisp, h, w = pred.shape
        disp_label = label_info[cfg.DISP.OUTPUT_PREFIX+'_0']
        semseg_label = label_info['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX, label_i)]
        disp_label  = F.adaptive_max_pool2d(disp_label, (h, w))
        match_label = torch.round(disp_label).long()
        match_label = match_label * semseg_label
        match_label[disp_label==0] = 255*cfg.DISP.MAX_DISPLACEMENT+1
        match_label[semseg_label==255] = 255*cfg.DISP.MAX_DISPLACEMENT+1
        pred = F.log_softmax(pred.view(bs, -1, h, w), dim =1)
        loss = self.crit_3d(pred, match_label)
        return loss

    def forward(self,data,**label_info):
        x, pred_semseg=self.pspnet(data)
        x=self.last(x)
        pred=self.glassGCN(x)
        return_dict = {}
        if self.training: # training
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            if not isinstance(pred, list):
                pred = [pred]
            for label_i, pred_i in enumerate(pred):
                # use 3D interp or not
                stride = 2**cfg.SEM.DOWNSAMPLE[label_i]
                pred_i = nn.functional.interpolate(pred_i, (cfg.DISP.MAX_DISPLACEMENT,
                    cfg.SEM.INPUT_SIZE[0]//stride, cfg.SEM.INPUT_SIZE[1]//stride), 
                    mode='trilinear',align_corners=False)
                bs, num_class, max_d, hei, wid = pred_i.shape
                #transfer 3D to 2D maps
                pred_disp = self.conv_last_disp(pred_i)
                pred_segdisp = self.conv_last_semseg(pred_i)
                # 3D softmax loss
                pred_3d = self.conv_last_segdisp(pred_i)
                loss = self.segdisp_loss(pred_3d, label_info, label_i)
                return_dict['losses']['loss_3d_%d'%label_i] = loss
                #pred_disp
                pred_disp = pred_disp.squeeze(1)
                loss, epe = self.disparityregression(pred_disp, pred_semseg, label_info, label_i=label_i, shape=(hei, wid))
                return_dict['losses']['loss_epe_%d'%label_i] = cfg.SEM.DEEP_SUB_SCALE[label_i]*loss
                return_dict['metrics']['epe_%d'%label_i] = epe
                #pred semseg
                pred_segdisp = pred_segdisp.squeeze(2)
                pred_semseg = pred_semseg + pred_segdisp
                pred_semseg = nn.functional.log_softmax(pred_semseg, dim=1)
                semseg_label = label_info['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX, label_i)]
                loss = self.crit(pred_semseg, semseg_label)
                return_dict['losses']['loss_semseg_%d'%label_i] = cfg.SEM.DEEP_SUB_SCALE[label_i]*loss
                return_dict['metrics']['acc_%d'%label_i] = self.pixel_acc(pred_semseg, semseg_label)
            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else: # inference
            pred_semseg, pred_disp = self.segdisp_pred(pred)
            return_dict['pred_semseg'] = pred_semseg
            return_dict['pred_disp'] = pred_disp
        return return_dict


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post


class NetStructure(DispSeg):
    def __init__(self):
        super(NetStructure,self).__init__()
        
    def forward(self,data):
        x=self.pspnet(data)
        pred=self.glassGCN(x)
        return pred
