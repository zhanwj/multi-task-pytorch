from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.nn import SynchronizedBatchNorm2d
from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
#from model.roi_crop.functions.roi_crop import RoICropFunction
#from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import modeling.semseg_heads as semseg_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import cv2
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
        valid = (label!=255).float()
        key_label = (label!=0).float() * valid
        acc_sum = (preds == label).float()
        pixel_sum = torch.sum(valid)
        key_sum = torch.sum(key_label)
        key_acc = torch.sum(acc_sum*key_label) / (key_sum + 1e-10)
        acc = torch.sum(acc_sum) / (pixel_sum + 1e-10)
        #if key_acc >= 0.3:
        #    cv2.imwrite('label.png',25*preds[0].detach().cpu().numpy())
        return [acc, key_acc]


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
        self.deep_sup_scale = cfg.SEM.DEEP_SUB_SCALE
        self.flag = 0
        #pos_weight=torch.ones(cfg.MODEL.NUM_CLASSES)
        #pos_weight[0] = cfg.SEM.OHEM_POS
        #self.crit = nn.NLLLoss(ignore_index=255, weight=Variable(pos_weight, requires_grad=False).to('cuda'))
        self.crit = nn.NLLLoss(ignore_index=255)

        if not cfg.SEM.OHEM_ON:
            self.loss_semseg = self.loss_norm
        else:
            self.loss_semseg = self.loss_ohem


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                print ('freeze_bn')
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
                print ('freeze_bn')

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            print ('loading weights for ResNet')
            resnet_utils.load_pretrained_imagenet_weights(self)
            print ('loading weights is done')

        if cfg.TRAIN.FREEZE_CONV_BODY:
            print ('freeze train conv_body')
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def loss_norm(self, pred_semseg, semseg_label, semseg_weight=None):
        if pred_semseg.shape[-1] != semseg_label.shape[-1]:
            pred_semseg=nn.functional.interpolate(pred_semseg,size=semseg_label.shape[1:],mode='bilinear',align_corners=False)
        pred_semseg = F.log_softmax(pred_semseg, dim=1)
        #semseg_weight = (semseg_label!=0).float() + cfg.SEM.OHEM_POS*(semseg_label==0).float()
        #semseg_weight = semseg_weight * (semseg_label!=255).float()
        loss = self.crit(pred_semseg, semseg_label)
        #loss = loss * semseg_weight
        #loss = torch.sum(loss) / torch.sum(semseg_weight)
        acc = self.pixel_acc(pred_semseg, semseg_label)
        return loss, acc

    def loss_ohem(self, pred_semseg, semseg_label):
        if pred_semseg.shape[-1] != semseg_label.shape[-1]:
            pred_semseg=nn.functional.interpolate(pred_semseg,size=semseg_label.shape[1:],mode='bilinear',align_corners=False)
        b, c, h, w = pred_semseg.shape
        scan = torch.arange(cfg.MODEL.NUM_CLASSES).view(1, cfg.MODEL.NUM_CLASSES, 1, 1).cuda()
        scan_pos = scan.repeat(b,1,1,1)
        scan_pos = (semseg_label.unsqueeze(1) == scan_pos).float()
        prob_pos = torch.softmax(pred_semseg, dim=1)
        prob_pos = torch.sum(prob_pos*scan_pos, dim=1)
        eval_label = (prob_pos<cfg.SEM.OHEM_POS).float()*(semseg_label!=255).float()
        pred_semseg = F.log_softmax(pred_semseg, dim=1)
        loss = self.crit(pred_semseg, semseg_label)
        loss = loss * eval_label
        loss = torch.sum(loss) / torch.sum(eval_label+1e-10)
        acc = self.pixel_acc(pred_semseg, semseg_label)
        return loss, acc

    def forward(self, data, **feed_dict):
        return_dict = {}
        if self.training: # training
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            if cfg.SEM.DECODER_TYPE.endswith('deepsup') or cfg.SEM.DECODER_TYPE.startswith('spn'): # use deep supervision technique
                (pred_semseg, pred_deepsup) = self.decoder(self.encoder(data, return_feature_maps=True))
                if not isinstance(pred_deepsup, list):
                    pred_deepsup = [pred_deepsup]
            else:
                pred_semseg = self.decoder(self.encoder(data, return_feature_maps=True))
            loss, acc = self.loss_semseg(pred_semseg, feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX,0)])
            return_dict['losses']['loss_semseg'] = loss
            return_dict['metrics']['accuracy_pixel'] = acc[0]
            return_dict['metrics']['key_accuracy_pixel'] = acc[1]
            if cfg.SEM.DECODER_TYPE.endswith('deepsup') or cfg.SEM.DECODER_TYPE.startswith('spn'):
                assert len(cfg.SEM.DEEP_SUB_SCALE)>1, 'per weight in each output side need to give'
                for i in range(1, len(cfg.SEM.DEEP_SUB_SCALE)):
                    loss_deepsup, _ = self.loss_semseg(pred_deepsup[i-1], 
                        feed_dict['{}_{}'.format(cfg.SEM.OUTPUT_PREFIX, 0)])
                    loss = loss + loss_deepsup * cfg.SEM.DEEP_SUB_SCALE[i]
            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else: # inference
             pred = self.decoder(self.encoder(data, return_feature_maps=True), segSize=cfg.SEM.INPUT_SIZE)
             assert pred.shape[1]==cfg.MODEL.NUM_CLASSES, 'need to predict for all class'
             assert pred.shape[-1]==cfg.SEM.INPUT_SIZE[-1] and pred.shape[-2]==cfg.SEM.INPUT_SIZE[-2], 'need to output full size in semseg_heads'
             assert torch.sum(pred<0)==0, 'need to output softmax in semseg_heads'
             return_dict['pred_semseg']=pred

        return return_dict



    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value

