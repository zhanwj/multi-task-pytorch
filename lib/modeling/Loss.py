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
import modeling.dispSeg_heads as semseg_heads
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
import numpy as np
logger = logging.getLogger(__name__)
netStructure=SummaryWriter(log_dir='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/network_structure/dispSeg/')


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

    def pixel_acc(self, preds, label):
        label = label.view(-1).contiguous()
        valid = (label != 255).long()
        acc_sum = torch.sum(valid*(preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

class disparityregression_bat(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()
        maxdisp = cfg.DISP.MAX_DISPLACEMENT
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp])).cuda(), requires_grad=False)
        self.zeros = Variable(torch.Tensor(np.zeros(np.array(range(maxdisp)),[1,maxdisp])).cuda(), requires_grad=False)
        def forward(self, x, semseg):
            zeros = self.zeros
            for i in range(x.size()[0]):
                zeros = torch.cat([zeros, self.zeros.repeat(1,semseg[i])],dim=0)
            disp = self.disp.repeat(x.size()[0],1)
            scans = torch.cat([zeros, disp], dim=1)
            for i in range(x.size()[0]):
                zeros = torch.cat([zeros, self.zeros.repeat(1,(cfg.MODEL.NUM_CLASSES-semseg[i]))],dim=0)
            scans = torch.cat([disp, zeros], dim=1)
            out = torch.sum(x*disp,1)
            return out

class disparityregression(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()
        maxdisp = cfg.DISP.MAX_DISPLACEMENT
        self.disp= Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp])).cuda(), requires_grad=False)
    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],cfg.MODEL.NUM_CLASSES)
        out = torch.sum(x*disp,1)
        return out

class Loss(SegmentationModuleBase):
    def __init__(self):
        super(Loss, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.iter=0
        self.draw=False

        ##define 3DSD loss
        self.crit = nn.NLLLoss(ignore_index=255*cfg.DISP.MAX_DISPLACEMENT+1)
        self.SmoothL1Loss = nn.SmoothL1Loss(reduction='none') #elementwise_mean

    def segdisp_loss(self, pred, label_info, lbl_i):
        semseg_label = label_info['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX, lbl_i)].long()
        disp_label   = label_info['{}_{}'.format(cfg.DISP.OUTPUT_PREFIX, lbl_i)]
        h, w = semseg_label.shape[-2:]
        disp_label = F.adaptive_max_pool2d(disp_label, (h, w))
        disp_label_match = torch.round(disp_label).long()
        assert torch.sum(disp_label_match>cfg.DISP.MAX_DISPLACEMENT)==0, 'errors in disp match'
        hybrid_label = semseg_label*cfg.DISP.MAX_DISPLACEMENT + disp_label_match
        hybrid_label[semseg_label==255] = 255 * cfg.DISP.MAX_DISPLACEMENT+1
        pred = torch.log(pred)
        loss = self.crit(pred, hybrid_label.view(-1).contiguous())
        return loss

    def segdisp_pred(self, pred, label_info, lbl_i=0):
        #using 3D softmax for predict
        #The shape of pred is (b*h*w, num_classes * max_disp)
        max_ids = torch.argmax(pred, dim=1)
        semseg_pred = max_ids // cfg.DISP.MAX_DISPLACEMENT
        num_sample = semseg_pred.shape[0]
        disp_scans = label_info['disp_scans'].repeat(pred.shape[0]//cfg.TRAIN.IMS_PER_BATCH, cfg.MODEL.NUM_CLASSES)
        disp_pred = torch.sum(disp_scans * pred, dim=1)
        #scans = torch.stack([semseg_pred*cfg.DISP.MAX_DISPLACEMENT + i for i in range(1, cfg.DISP.MAX_DISPLACEMENT+1)], dim=1)
        #disp_pred = pred[scans]
        return semseg_pred, disp_pred


    def disp_loss(self, pred, targe, sparse=True, shape=(30,30)):
        if sparse:
            targe = F.adaptive_max_pool2d(targe, shape)
        targe=targe.view(-1).contiguous()
        EPE_map = self.SmoothL1Loss(pred, targe)
        positive = (targe > 0).long()
        EPE_map = EPE_map[positive]
        loss = EPE_map.mean()
        return loss

    def forward(self,pred,**label_info):
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
                _, num_class, max_d, hei, wid = pred_i.shape
                #transfer 3D to 2D maps
                pred_i = pred_i.permute(0,3,4,1,2)
                pred_i = pred_i.view(-1, hei*wid, num_class * max_d).contiguous()
                pred_i = pred_i.view(-1, num_class * max_d).contiguous()
                pred_i = nn.functional.softmax(pred_i, dim=1)
                #3D sematic disparity loss
                loss = self.segdisp_loss(pred_i, label_info, label_i)
                return_dict['losses']['loss_segdisp'] = loss
                #stat for predict
                pred_semseg, pred_disp = self.segdisp_pred(pred_i, label_info, label_i)
                return_dict['disp_image']=pred_disp
                return_dict['semseg_image']=pred_semseg
                return_dict['metrics']['pixel_accuary'] = \
                        self.pixel_acc(pred_semseg, label_info['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX, label_i)])
                return_dict['losses']['loss_epe'] = \
                        self.disp_loss(pred_disp, label_info['{}_{}'.format(cfg.DISP.OUTPUT_PREFIX, label_i)],
                                shape=(hei, wid))
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


    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv


    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
