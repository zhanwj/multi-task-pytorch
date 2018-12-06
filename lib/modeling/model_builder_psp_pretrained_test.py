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
import modeling.semseg_heads as semseg_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils

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
        valid = (label != 255).long()
        acc_sum = torch.sum((preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class Generalized_SEMSEG(SegmentationModuleBase):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        #define encoder 
        builder = semseg_heads.ModelBuilder()
        self.encoder = builder.build_encoder(
            arch=cfg.SEM.ARCH_ENCODER,
            fc_dim=cfg.SEM.FC_DIM)

        #define semseg heads
        self.decoder = builder.build_decoder(
                arch=cfg.SEM.DECODER_TYPE,
                fc_dim=cfg.SEM.FC_DIM,
                num_class=cfg.MODEL.NUM_CLASSES,
                use_softmax=not self.training,
                weights='')
        self.crit = nn.NLLLoss(ignore_index=255)
        self.deep_sup_scale = cfg.SEM.DEEP_SUB_SCALE
        self.flag = 0
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
                pred = self.decoder(self.encoder(data, return_feature_maps=True))
            if cfg.SEM.DECODER_TYPE.endswith('deepsup') and not isinstance(pred_deepsup, list):
                pred_deepsup = [pred_deepsup]
            pred_semseg = nn.functional.log_softmax(pred, dim=1)
            pred_semseg=nn.functional.interpolate(pred_semseg,size=cfg.SEM.INPUT_SIZE,mode='bilinear',align_corners=False)
            loss = self.crit(pred_semseg, feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX,0)])
            return_dict['losses']['loss_semseg'] = loss
            acc = self.pixel_acc(pred_semseg, feed_dict[cfg.SEM.OUTPUT_PREFIX+'_0'])
            return_dict['metrics']['accuracy_pixel'] = acc
            if cfg.SEM.DECODER_TYPE.endswith('deepsup'):
                for i in range(1, len(cfg.SEM.DOWNSAMPLE)):
                    loss_deepsup = self.loss_semseg(pred_deepsup[i-1], 
                        feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX, i)])
                    loss = loss + loss_deepsup * self.deep_sup_scale[i]
            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else: # inference
            if cfg.SEM.DECODER_TYPE.endswith('deepsup'): # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(data, return_feature_maps=True))
                return_dict['pred_semseg']=pred
                return_dict['pred_deepsup']=pred_deepsup
            else:
                pred = self.decoder(self.encoder(data, return_feature_maps=True))
                return_dict['pred_semseg']=pred
            if cfg.SEM.DECODER_TYPE.endswith('deepsup') and not isinstance(pred_deepsup, list):
                pred_deepsup = [pred_deepsup]
            #pred = self.decoder(self.encoder(data, return_feature_maps=True))
            #return_dict['pred_semseg'] = pred
            #return_dict['pred_semseg']=pred
            #return_dict['pred_deepsup']=pred_deepsup
        return return_dict


    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv


    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
