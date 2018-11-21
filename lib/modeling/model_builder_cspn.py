from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import modeling.dispSeg_heads_cspn as semseg_heads_cspn
import modeling.dispSeg_heads_1x as semseg_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.Lossfuction as Lossfuction
import utils.resnet_weights_helper as resnet_utils
from lib.nn import SynchronizedBatchNorm2d
import pynvml
import cv2
import modeling.CRL as CRL
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from torch.utils.checkpoint import checkpoint
logger = logging.getLogger(__name__)


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
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
        self.iter = 0

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label != 255).long()
        acc_sum = torch.sum(valid*(preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class Generalized_SEGDISP(SegmentationModuleBase):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()
        print("hello")

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.iter=0
        self.draw=False
        builder = semseg_heads.ModelBuilder()

        #define encoder 
        if cfg.SEM.USE_RESNET:

            self.encoder = get_func(cfg.MODEL.CONV_BODY)()
        else:
            builder = semseg_heads.ModelBuilder()
            self.encoder = builder.build_encoder(
                arch=cfg.SEM.ARCH_ENCODER,
                fc_dim=cfg.SEM.FC_DIM,
                weights=cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS
                )

        #define shape weights
        self.decoder = builder.build_decoder(
                arch=cfg.SEM.DECODER_TYPE,
                fc_dim=cfg.SEM.FC_DIM,
                num_class=cfg.MODEL.NUM_CLASSES,
                use_softmax=not self.training,
                weights='')

        self.minicspn = semseg_heads.MiniCSPN(512)

        self.crit = nn.NLLLoss(ignore_index=255)
                
        self.SmoothL1Loss = nn.SmoothL1Loss(reduction='none') #elementwise_mean
      

    def disp_loss(self, disp_pred, label_info, label_i=0, sparse=True):
        targe = label_info['disp_label_0'].unsqueeze(1)
        shape = disp_pred.shape[-2:]
        if sparse:
            targe = F.adaptive_max_pool2d(targe, shape)
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



    def _init_modules(self):

        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def _init_weights_kaiming(self,m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_uniform_(m.weight)
        if type(m) == nn.ConvTranspose2d:
            nn.init.kaiming_uniform_(m.weight)


    def _init_weights_normal(self,m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.01)

    def forward(self,data,**label_info):
        left=data[0:cfg.TRAIN.IMS_PER_BATCH,:,:,:]
        right=data[cfg.TRAIN.IMS_PER_BATCH:,:,:,:]
        #print("forward start")
        return_dict = {}
        #print (feed_dict['semseg_label_1'][0,0,0])
        if self.training: # training
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            #build network
            resFeature = self.encoder(data,return_feature_maps=True)
            asppFeature=self.decoder(resFeature)
            predict_items = self.minicspn(asppFeature[0], resFeature, left, right)
            disp_outside, semseg_outside = predict_items

            #disp_outside, from 0 to N, is ratio inverst
            return_dict['losses']['loss_disp'] = 0
            for ids, disp_i in enumerate(disp_outside):
                loss_disp, epe = self.disp_loss(disp_i, label_info)
                return_dict['losses']['loss_disp'] += cfg.DISP.DEEP_SUB_SCALE[ids] * loss_disp
                return_dict['metrics']['pixel_epe'] = epe
            disp_image=''
            semseg_image=''
            #semseg_outside, 0: res4s 1: full size
            semseg_outside[0] = F.log_softmax(semseg_outside[0], dim=1)
            semseg_outside[1] = F.log_softmax(semseg_outside[1], dim=1)
            return_dict['losses']['loss_semseg'] = self.crit(semseg_outside[1], label_info['%s_0'%cfg.SEM.OUTPUT_PREFIX])
            return_dict['losses']['loss_semseg'] += \
                    cfg.SEM.DEEP_SUB_SCALE[1] * self.crit(semseg_outside[0], label_info['%s_1'%cfg.SEM.OUTPUT_PREFIX])
            acc = self.pixel_acc(semseg_outside[1], label_info['%s_0'%cfg.SEM.OUTPUT_PREFIX])
            return_dict['metrics']['accuracy_pixel'] = acc
            return_dict['disp_image'] = disp_image
            return_dict['semseg_image']= semseg_image

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else: # inference
            pred = self.decoder(self.encoder(data, return_feature_maps=True), segSize=segSize)
            pred_semseg, pred_disp = torch.split(pred, 1, dim=0)
            return_dict['pred_semseg'] = pred_semseg
            return_dict['pred_disp'] = pred_disp

        return return_dict


    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv


    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
