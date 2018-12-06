from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_,constant_
from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import modeling.semseg_heads as semseg_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import modeling.fcn8s as fcn 
import modeling.spn_bat as spn
import modeling.pspnet_dispSeg as psp
logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum((preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class Generalized_SEMSPN(SegmentationModuleBase):
    def __init__(self):
        super(Generalized_SEMSPN, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        self.pspnet=psp.PSPNET()
        self.spn_guidance = fcn.FCN8s()
        self.spn_net =spn.SPN()
        self.crit = nn.NLLLoss(ignore_index=255)
        self.deep_sup_scale = cfg.SEM.DEEP_SUB_SCALE


        self.guide_conv1=nn.Conv2d(cfg.MODEL.NUM_CLASSES,32,kernel_size=3,padding=1,stride=2,bias=False)
        self.guide_conv2=nn.Conv2d(256,384,kernel_size=3,padding=1,stride=1,bias=False)

        self.elt_resize_deconv=nn.Sequential(                      #1/2
            nn.ConvTranspose2d(32,32,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Conv2d(32,64,3,padding=1,stride=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,cfg.MODEL.NUM_CLASSES,kernel_size=3,padding=1,stride=1,bias=False)
            )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                kaiming_normal_(m.weight, 0.01)
                if m.bias is not None:
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

        if cfg.SEM.SPN_PRETRAINED is not None:
            print("loading spn weights")
            state_dict={}
            pretrained=torch.load(cfg.SEM.SPN_PRETRAINED, map_location=lambda storage, loc: storage)
            #pretrained = pretrained['model']
            self.spn_guidance.load_state_dict(pretrained,strict=False)
            print("weights load success")

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, **feed_dict):
        return_dict = {}
        if self.training: # training
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            if cfg.SEM.DECODER_TYPE.endswith('deepsup'): # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(data, return_feature_maps=True))
            else:
                pred = self.pspnet(data)
            if cfg.SEM.DECODER_TYPE.endswith('deepsup') and not isinstance(pred_deepsup, list):
                pred_deepsup = [pred_deepsup]
            pred_semseg=nn.functional.interpolate(pred,scale_factor=8,mode='bilinear',align_corners=False)  #1/2
            acc = self.pixel_acc(pred_semseg, feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX,0)])
            return_dict['metrics']['accuracy_pixel'] = acc
            #print("pred_semseg:",pred_semseg.shape)
            #spn_data=nn.functional.interpolate(data,scale_factor=0.5,mode='bilinear',align_corners=False)
            #print("spn_data:",spn_data.shape)
            #spn scan
            
            if cfg.SEM.SPN_CROPSIZE[0]!=cfg.SEM.INPUT_SIZE[0]:
                hei, wid=cfg.SEM.SPN_CROPSIZE
                crop_h = torch.round(torch.rand(1)*(0.5*cfg.SEM.INPUT_SIZE[0]-hei)).long()
                crop_w = torch.round(torch.rand(1)*(0.5*cfg.SEM.INPUT_SIZE[1]-wid)).long()
                spn_guidance = self.spn_guidance(data[:, :, crop_h:crop_h+hei, crop_w:crop_w+wid])
                pred_semseg=pred_semseg[:, :, crop_h:crop_h+hei, crop_w:crop_w+wid]                     #1/4
                pred_semseg = self.spn_net(pred_semseg, spn_guidance)
                #pred_semseg=nn.functional.interpolate(pred_semseg,scale_factor=2,mode='bilinear',align_corners=False)  #1/2
                
                
                #final semseg loss
                crop_h = crop_h * 2
                crop_w = crop_w * 2
                wid = wid * 2
                hei = hei * 2
                semseg_label = feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX,0)][:, crop_h:crop_h+hei, crop_w:crop_w+wid]
            else:
                spn_guidance = self.spn_guidance(data)
                
                pred_semseg = self.spn_net(pred_semseg, spn_guidance)
                #pred_semseg=self.elt_resize_deconv(pred_semseg)
                #pred_semseg=nn.functional.interpolate(pred_semseg,scale_factor=2,mode='bilinear',align_corners=False)
                semseg_label = feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX,0)]
                #semseg_label = nn.functional.interpolate(semseg_label,scale_factor=0.5,mode='bilinear',align_corners=False)
            
            pred_semseg = nn.functional.log_softmax(pred_semseg, dim=1)
            loss = self.crit(pred_semseg, semseg_label)
            return_dict['losses']['loss_semseg'] = loss
            if cfg.SEM.DECODER_TYPE.endswith('deepsup'):
                for i in range(1, len(cfg.SEM.DOWNSAMPLE)):
                    loss_deepsup = self.crit(pred_deepsup[i-1], 
                        feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX, i)])
                    loss = loss + loss_deepsup * self.deep_sup_scale[i]
            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else: # inference
            pred = self.decoder(self.encoder(data, return_feature_maps=True), segSize=segSize)
            return_dict['pred_semseg'] = pred

        return return_dict


    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv


    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
