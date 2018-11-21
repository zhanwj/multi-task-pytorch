
import torch
import torch.nn as nn
import torchvision
from . import resnet as resnet
from . import resnext as resnext
from torch.nn.init import kaiming_normal_,constant_,normal_
from core.config import  cfg
import torch.nn.functional as F
import modeling.CRL as CRL
import modeling.cspn as cspn
import time
timer=time.time

if not cfg.SEM.BN_LEARN:
    from lib.nn import SynchronizedBatchNorm2d
else:
    import torch.nn.BatchNorm2d as SynchronizedBatchNorm2d

def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)

class CorrelationLayer1D(nn.Module):
    def __init__(self, max_disp=40, stride_2=1):
        super(CorrelationLayer1D, self).__init__()
        self.max_displacement = max_disp
        self.stride_2 = stride_2

    def forward(self, x_1, x_2):
        x_1 = x_1
        x_2 = F.pad(x_2, (int(self.max_displacement*0.2),int(self.max_displacement*0.8), 0, 0))
        return torch.cat([torch.sum(x_1 * x_2[:, :, :, _y:_y + x_1.size(3)], 1).unsqueeze(1) for _y in
                          range(0, self.max_displacement +1, self.stride_2)], 1)

class CorrelationLayer1DMinus(nn.Module):
    def __init__(self, max_disp=40, stride_2=1):
        super(CorrelationLayer1DMinus, self).__init__()
        self.max_displacement = max_disp
        self.stride_2 = stride_2
    def forward(self, x_1, x_2):
        x_1 = x_1
        ee=0.000001
        x_2 = F.pad(x_2, (int(self.max_displacement*0.2),int(self.max_displacement*0.8), 0, 0))
        minus=torch.cat([torch.sum(x_1 - x_2[:, :, :, _y:_y + x_1.size(3)], 1).unsqueeze(1) for _y in
                          range(0, self.max_displacement +1, self.stride_2)], 1)
        inverse=1/(minus+ee)
        return torch.sigmoid_(inverse)

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

class CorrelationLayerCosineSimilarity(nn.Module):
    def __init__(self, max_disp=40, stride_2=1,dim=1,eps=1e-6):
        super(CorrelationLayerCosineSimilarity, self).__init__()
        self.max_displacement = max_disp
        self.stride_2 = stride_2
        self.cos=torch.nn.CosineSimilarity(dim=1,eps=1e-6)
    def forward(self, x_1, x_2):
        x_1 = x_1
        x_2 = F.pad(x_2, (int(self.max_displacement*0),int(self.max_displacement*1), 0, 0))
        similarity=torch.cat([self.cos(x_1 ,x_2[:, :, :, _y:_y + x_1.size(3)]).unsqueeze(1) for _y in
                          range(0, self.max_displacement +1, self.stride_2)], 1)
        return similarity

def costVolume2(leftFeature,rightFeature,max_displacement):
    cost = torch.zeros(leftFeature.size()[0], leftFeature.size()[1]*2, max_displacement,  leftFeature.size()[2],  leftFeature.size()[3]).cuda()

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

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
    def forward(self, feed_dict, *, segSize=None):
        if segSize is None: # training
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict[cfg.SEM.OUTPUT_PRIFEX+'_0'])
            if self.deep_sup_scale is not None:
                for i in range(2, len(cfg.SEM.DOWNSAMPLE)):
                    loss_deepsup = self.crit(pred_deepsup, 
                            feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PRIFEX, i)])
                    loss = loss + loss_deepsup * self.deep_sup_scale[i]

            acc = self.pixel_acc(pred, feed_dict[cfg.SEM.OUTPUT_PRIFEX+'_0'])
            return loss, acc
        else: # inference
            pred = self.decoder(self.encoder(feed_dict['data'], return_feature_maps=True), segSize=segSize)
            return pred


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18_dilated8':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet18_dilated16':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated8_3DConv':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated3DConv(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet

        elif arch == 'resnext101_dilated8':
            orig_resnet = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnext101_dilated8_64':
            orig_resnet = resnext.__dict__['resnext101_64'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        if arch == 'c1_bilinear_deepsup':
            net_decoder = C1BilinearDeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_bilinear':
            net_decoder = C1Bilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear_deepsup':
            net_decoder = PPMBilinearDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_bilinear3D':
            net_decoder = PPMBilinear3D(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        elif arch == 'upernet_tmp':
            net_decoder = UPerNetTmp(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.correlation=CorrelationLayer1D(max_disp=40,stride_2=1)
        self.conv_rdi = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True))                                                   
                
        self.conv_r = nn.Conv2d(357, 512, kernel_size=3, stride=1,padding=1, bias=False)
                             
        self.bn4=SynchronizedBatchNorm2d(512)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);  #256
        x = self.layer2(x); conv_out.append(x);  #512
        left, right=torch.split(x, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        corr=self.correlation(left,right)
        conv_rdi=self.conv_rdi(left)
        x =torch.cat((conv_rdi,corr),dim=1)
        x=self.relu2(self.bn4(self.conv_r(x)))
        x = torch.cat((left, x), dim=0)
        x = self.layer3(x); conv_out.append(x);  #1024
        x = self.layer4(x); conv_out.append(x);  #2048

        if return_feature_maps:
            return conv_out
        return [x]




    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x); conv_out.append(x);
        #print("layer1:",x.shape)
        x = self.layer2(x); conv_out.append(x);
        #print("layer2:",x.shape)
        

        left, right=torch.split(x, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        #print("left:",left.shape)
        #print("right:",right.shape)
        corr=self.correlation(left,right)
        #print("corr:",corr.shape)
        conv_rdi=self.conv_rdi(left)
        #print("conv_rdi:",conv_rdi.shape)
        x =torch.cat((conv_rdi,corr),dim=1)
        x=self.relu2(self.bn4(self.conv_r(x)))

        x = torch.cat((left, x), dim=0)



        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated3DConv(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8,max_displacement=40):
        super(ResnetDilated3DConv, self).__init__()
        from functools import partial
        self.max_displacement=max_displacement

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        if cfg.SEM.LAYER_FIXED:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.conv2.parameters():
                param.requires_grad =  False
            for param in self.conv3.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);
        if return_feature_maps:
            return conv_out
        return [x]



class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        if cfg.DISP.COST_VOLUME_TYPE == 'CorrelationLayer1D':
            self.correlation=CorrelationLayer1D(max_disp=40,stride_2=1)
        if cfg.DISP.COST_VOLUME_TYPE == 'CorrelationLayer1DMinus':
            self.correlation=CorrelationLayer1DMinus(max_disp=40,stride_2=1)
        if cfg.DISP.COST_VOLUME_TYPE =='CorrelationLayerCosineSimilarity':
            self.correlation=CorrelationLayerCosineSimilarity(max_disp=40)


        self.bn4=SynchronizedBatchNorm2d(512)

        self.conv_rdi = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True))
        self.conv_r = nn.Conv2d(297, 512, kernel_size=3, stride=1,padding=1, bias=False)

        if cfg.SEM.LAYER_FIXED:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.conv2.parameters():
                param.requires_grad =  False
            for param in self.conv3.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x); conv_out.append(x);
   
        x = self.layer2(x); conv_out.append(x);

        left, right=torch.split(x, cfg.TRAIN.IMS_PER_BATCH, dim=0)

        corr=self.correlation(left,right)

        conv_rdi=self.conv_rdi(left)
        x =torch.cat((conv_rdi,corr),dim=1)
        x=self.relu2(self.bn4(self.conv_r(x)))

        x = torch.cat((left, x), dim=0)

        x = self.layer3(x); conv_out.append(x);
     
        x = self.layer4(x); conv_out.append(x);
       
        if return_feature_maps:
            return conv_out
        return [x]


# last conv, bilinear upsample
class C1BilinearDeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling, bilinear upsample
class PPMBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, conv_out, segSize=None):
        if cfg.SEM.USE_RESNET:
            conv5=conv_out
        else:
            conv5 = conv_out[-1]
        #conv5=conv_out

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        return x


# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=1024,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                #SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        #self.reduce=nn.Conv2d(fc_dim*2,fc_dim,kernel_size=1,stride=1,padding=0,bias=False)
        #self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.aspp_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        #self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        #self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        
        if cfg.SEM.USE_RESNET:
            conv5=conv_out
        else:
            conv5 = conv_out[-1]
        #conv_out, 2, c, h, w, dim 0 is semseg and disp
        input_size = conv5.size()
        semseg_conv, disp_conv = torch.split(conv5, input_size[0]//2 ,dim=0)
        #conv5 is 1, 2*c, h, w
        conv5 = torch.cat([semseg_conv, disp_conv], dim=1)
        #conv5=self.reduce(conv5)
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.aspp_last(ppm_out)



        # deep sup
        conv4 = conv_out[-2]
        #_ = self.cbr_deepsup(conv4)
        #_ = self.dropout_deepsup(_)
        #_ = self.conv_last_deepsup(_)

        #X = nn.functional.log_softmax(x, dim=1)
        #_ = nn.functional.log_softmax(_, dim=1)
        return [x, conv4]


class PPMBilinear3D(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),channelsReduction=19):
        super(PPMBilinear3D, self).__init__()
        self.use_softmax = use_softmax
        self.channelsReduction=channelsReduction
        self.ppm = []
        self.width=96
        self.height=96
        self.semseg=cfg.MODEL.NUM_CLASSES
        self.max_displacement=cfg.DISP.FEATURE_MAX_DISPLACEMENT
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
        
        cost_channels = channelsReduction*2
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
            nn.BatchNorm2d(channelsReduction)
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
            nn.Conv3d(inplanes,planes,kernel_size=3,stride=stride1,groups=groups,padding=1,bias=False),
            nn.BatchNorm3d(planes),
            nn.Conv3d(planes,planes,kernel_size=3,stride=stride2,groups=groups,padding=1,bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
            )
    def _Deconv3D(self,inplanes,planes,kernel_size=3,stride=2,padding=1,out_padding=1,groups=19,bias=False):
            return nn.ConvTranspose3d(inplanes,planes,kernel_size,stride,padding,out_padding,groups=groups,bias=bias)


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
        x = self.reduce(x)
        left, right=torch.split(x, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        cost = costVolume2(left,right,cfg.DISP.FEATURE_MAX_DISPLACEMENT)
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

class GCNASPP(nn.Module):
    def __init__(self,inplanes,planes,d,h,w,scale,pool_scales=(2,4,8,16)):
        super(GCNASPP,self).__init__()
        self.inplanes=inplanes
        self.planes=planes
        self.semsegNums=19
        self.disparity=self._Conv3d(self.inplanes,self.planes,kernel_size=(11,1,1),padding=(5,0,0))
        self.width=self._Conv3d(self.inplanes,self.planes,kernel_size=(1,1,11),padding=(0,0,5))
        self.height=self._Conv3d(self.inplanes,self.planes,kernel_size=(1,11,1),padding=(0,5,0))

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(scale),
                nn.Conv3d(self.semsegNums,self.semsegNums,kernel_size=1,bias=False),
                nn.BatchNorm3d(self.semsegNums),
                nn.ReLU(inplace=True)
                ))
        self.ppm = nn.ModuleList(self.ppm)

        self.aspp_last = nn.Sequential(
            nn.Conv3d(5*self.semsegNums,self.semsegNums,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm3d(self.semsegNums),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
            )

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
                constant_(m.weight,1)
                constant_(m.bias,0)


    def _Conv3d(self,inplanes,planes,kernel_size,stride=1,groups=1,padding=1):
        return nn.Sequential(
            nn.Conv3d(inplanes,planes,kernel_size,stride,padding=padding,bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        disparity=self.disparity(x)
        width = self.width(x)
        height = self.height(x)
        out=disparity+width+height
        input_size = (out).size()
        ppm_out=[out]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(out),(input_size[2],input_size[3],input_size[4]),
                mode='trilinear',align_corners=False
                ))

        ppm_out=torch.cat(ppm_out,1)
        out = self.aspp_last(ppm_out)
        return out

# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)



class MiniPSMNet(nn.Module):
    def __init__(self):
        super(MiniPSMNet,self).__init__()
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
        self.stack1_3 = self._createStack(cost_channels*4,cost_channels*8)
        self.stack2_1 = self._Deconv3D(cost_channels*8,cost_channels*4)
        self.stack2_2 = self._Deconv3D(cost_channels*4,cost_channels*2)
        self.stack2_3 = self._Deconv3D(cost_channels*2,cost_channels)

        
        
        
        self.to2D = nn.Conv3d(cost_channels,1,kernel_size=1,strid=1)
        self.reduce = self._ruduce2D(512,self.channelsReduction)
        self.predict=self._predict(cost_channels)
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

    def _ruduce2D(self,inplanes,planes):
        return nn.Sequential(
            nn.Conv2d(inplanes,planes,kernel_size=1,strid=1),
            nn.Conv2d(planes,planes,kernel_size=3,strid=1,padding=1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
            )
    def _predict(self,inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes,1,kernel_size=1,strid=1),
            nn.ReLU(inplace=True)
            )

    def forward(self, conv_out):
        x = self.reduce(conv_out)
        left, right=torch.split(x, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        cost = costVolume2(left,right,self.max_displacement)

        stack0=self.stack0(cost)
        stack1=self.stack1(stack0)

        stack1_1=self.stack1_1(stack1)
        stack1_2=self.stack1_2(stack1_1)
        stack1_3=self.stack1_3(stack1_2)
    
        stack2_1=self.stack2_1(stack1_3)+stack1_2
        stack2_2=self.stack2_2(stack2_1)+stack1_1
        stack2_3=self.stack2_3(stack2_2)+stack1
        
        out2d=self.to2D(stack2_3)
        out=torch.squeeze(out2d,dim=1)
        predict = self.predict(out)
        return [out,predict]

class TConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1):
        super(TConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=False)

    def forward(self, x):
        return F.leaky_relu(self.conv.forward(x), negative_slope=0.1, inplace=True)

class FusionNet(nn.Module):
    def __init__(self,inplanes):
        super(FusionNet,self).__init__()
        self.out_channels=32
        self.rdi = nn.Conv2d(512+cfg.SEM.SD_DIM*2,self.out_channels*8)

        self.upconv8_4 = self._TConv(self.out_channels*8,self.out_channels*4)
        self.upconv4_2 = self._TConv(self.out_channels*4,self.out_channels*2)
        self.upconv2_1 = self._TConv(self.out_channels*2,self.out_channels)

        self.pr8 = nn.Conv2d(self.out_channels*8,1,kernel_size=3,strid=1,padding=1,bias=False)  #512
        self.pr4 = nn.Conv2d(self.out_channels*4,1,kernel_size=3,strid=1,padding=1,bias=False)  #256
        self.pr2 = nn.Conv2d(self.out_channels*2,1,kernel_size=3,strid=1,padding=1,bias=False)  #128
        self.pr1 = nn.Conv2d(self.out_channels,1,kernel_size=3,strid=1,padding=1,bias=False)    #64

        self.fusion8=self._fusion(512+512+cfg.SEM.SD_DIM*2,self.out_channels*8)
        self.fusion4=self._fusion(self.out_channels*4+256,self.out_channels*4)
        self.fusion2=self._fusion(self.out_channels*2+128,self.out_channels*2)
        self.fusion1=self._fusion(self.out_channels*1,self.out_channels)





    def _Tconv(self,inplanes,planes):
        return nn.Sequential(
            nn.ConvTranspose2d(inplanes,planes,kernel_size=3,strid=2,padding=1),
            nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(negative_slope=0.1,inplace=True)
            )

    def _fusion(self,inplanes,planes,kernel_size=3,stride=1,padding=1):
        return nn.Sequential(
            nn.Conv2d(inplanes,planes,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(negative_slope=0.1,inplace=True))

    def forward(self,semdisp,psm,resFeature):
        pred_semseg, pred_disp = torch.split(pred, cfg.TRAIN.IMS_PER_BATCH, dim=0)

        conv1a, _ = torch.split(FeatureMap[0], cfg.TRAIN.IMS_PER_BATCH, dim=0)    #64channels
        #_ , conv1a = torch.split(conv1a, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        conv2a, _ = torch.split(FeatureMap[1], cfg.TRAIN.IMS_PER_BATCH, dim=0)   #128channels
        #_ , conv2a =  torch.split(conv2a, cfg.TRAIN.IMS_PER_BATCH, dim=0)
        _, layer4 = torch.split(FeatureMap[4], cfg.TRAIN.IMS_PER_BATCH, dim=0)

        feature8 = self.fusion8(torch.cat((pred_disp,psm,layer4),dim=1))
        pr8=self.pr8(feature8)
        upfeature8_4=self.upconv8_4(torch.cat(pr8,feature8),dim=1)

        feature4 = self.fusion4(torch.cat((upfeature8_4,conv2a),dim=1))
        pr4=self.pr4(feature4)
        upfeature4_2=self.upconv4_2(torch.cat(pr4,feature4),dim=1)

        feature2 = self.fusion2(torch.cat((upfeature4_2,conv1a),dim=1))
        pr2=self.pr2(feature2)
        upfeature2_1 =sefl.upconv2_1(torch.cat(pr2,feature2),dim=1)

        pr1=self.pr1(torch.cat(upfeature2_1),dim=1)

        return[pr1,pr2,pr4,pr8]


class MiniCSPN(nn.Module):
    def __init__(self,in_channels):
        super(MiniCSPN,self).__init__()
        self.in_channels=in_channels
        self.conv1_1=[]
        self.out=[]
        self.FupCat=[]
        for i in range(0,4):
            self.FupCat.append(Gudi_UpProj_Block_Cat(self.in_channels//2**i,self.in_channels//2**(i+1),
                                                  oheight=cfg.SEM.INPUT_SIZE[0]//2**(i+1),owidth=cfg.SEM.INPUT_SIZE[1]//2**(i+1)))
            self.out.append(self._out(self.in_channels//2**(i+1)))

        self.conv1_1=nn.ModuleList(self.conv1_1)
        self.FupCat=nn.ModuleList(self.FupCat)
        self.out=nn.ModuleList(self.out)

        self.pred_disp16x=self._out(self.in_channels)


        self.conv_last_smeseg=nn.Sequential(
            nn.Conv2d(self.in_channels+256,self.in_channels//2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(self.in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels//2,19,kernel_size=3,padding=1,bias=False))

        self.semseg_deepsup=nn.Conv2d(1024,19,kernel_size=3,padding=1,bias=False)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                constant_(m.weight,1)
                constant_(m.bias,0)
        self.disp_outside = []


    def _conv(self,inplanes,planes,kernel_size=3,stride=1,padding=1,bias=False):
        return nn.Sequential(
            nn.Conv2d(inplanes,planes,kernel_size,stride=stride,padding=padding,bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )
    
    def _semOut(self,inplanes,kernel_size=3,stride=1,padding=1,bias=False):
        return nn.Sequential(
            nn.Conv2d(inplanes,19,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias))

    def _out(self,inplanes,kernel_size=3,stride=1,padding=1,bias=False):
        return nn.Sequential(
            nn.Conv2d(inplanes,inplanes,kernel_size=kernel_size,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes,1,kernel_size=kernel_size,stride=1,padding=1,bias=True))


    def _up_pooling(self, x, scale_factor,mode='bilinear',oheight=0,owidth=0):
        if mode =='bilinear':
            return nn.functional.interpolate(x,scale_factor=scale_factor, mode='bilinear')
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if oheight !=0 and owidth !=0:
            x = x[:,:,0:oheight, 0:owidth]
        mask = torch.zeros_like(x)
        for h in range(0,oheight, 2):
            for w in range(0, owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x

    def forward(self,x,resFeature,left,right):
        layer1=resFeature[1]
        leftFeature,rightFeature=torch.split(layer1,cfg.TRAIN.IMS_PER_BATCH,dim=0)
        # disp decoder
        res4,_=torch.split(resFeature[-2],cfg.TRAIN.IMS_PER_BATCH,dim=0)
        self.disp_outside=[]
        dispNx_in = x
        disp16x=self.pred_disp16x(dispNx_in)
        self.disp_outside.append(disp16x)

        #use up_cat to encoder
        for i in range(0,4):
            #dispNx_in = self._up_pooling(dispNx_in, scale_factor=2)
            #dispNx_in = self.conv1_1[i](dispNx_in)  #reduce
            dispNx_in =self.FupCat[i](dispNx_in, left, right, ratio=4-i)
            self.disp_outside.append(self.out[i](dispNx_in))
        #decode for semseg
        semseg4x_features=self._up_pooling(x, scale_factor=4)
        semseg4x_features = torch.cat([semseg4x_features, leftFeature], dim=1)
        semseg4x_maps = self.conv_last_smeseg(semseg4x_features)
        semseg_maps = self._up_pooling(semseg4x_maps, scale_factor=4)
        semseg_res4=self.semseg_deepsup(res4)
        return  self.disp_outside, [semseg_res4, semseg_maps]



class Gudi_UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                kaiming_normal_(m.weight,0.1)
                if m.bias is not None:
                    constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                constant_(m.weight,1)
                constant_(m.bias,0)
        
    def _up_pooling(self, x, scale):
        
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight !=0 and self.owidth !=0:
            x = x[:,:,0:self.oheight, 0:self.owidth]
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x
    
    def forward(self, x):
        
        x = self._up_pooling(x, 2)
        
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        
        return out


class Gudi_UpProj_Block_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1_1 = nn.Conv2d(out_channels+6, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        
    def _up_pooling(self, x, scale,mode='bilinear',oheight=0,owidth=0):
        if mode =='bilinear':
            return  nn.functional.interpolate(x,scale_factor=scale, mode='bilinear')
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if oheight !=0 and owidth !=0:
            x = x[:,:,0:oheight, 0:owidth]
        mask = torch.zeros_like(x)
        for h in range(0,oheight, 2):
            for w in range(0, owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x
    
    def forward(self, x, left,right,ratio):
        left=left[:,:,::2**(ratio-1),::2**(ratio-1)]
        right=right[:,:,::2**(ratio-1),::2**(ratio-1)]
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, left,right), 1)
        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out

class OriginalGudi_UpProj_Block_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(OriginalGudi_UpProj_Block_Cat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1_1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.oheight = oheight
        self.owidth = owidth
        
    def _up_pooling(self, x, scale):
        
        x = nn.Upsample(scale_factor=scale, mode='nearest')(x)
        if self.oheight !=0 and self.owidth !=0:
            x = x[:,:,0:self.oheight, 0:self.owidth]
        mask = torch.zeros_like(x)
        for h in range(0, self.oheight, 2):
            for w in range(0, self.owidth, 2):
                mask[:,:,h,w] = 1
        x = torch.mul(mask, x)
        return x
    
    def forward(self, x, side_input):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, side_input), 1)
        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out




