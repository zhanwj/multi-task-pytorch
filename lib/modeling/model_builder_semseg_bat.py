from functools import wraps
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.nn import SynchronizedBatchNorm2d
from core.config import cfg
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
        valid = (label!=255).long()
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
        self.deep_sup_scale = cfg.SEM.DEEP_SUB_SCALE
        self.flag = 0
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

    def loss_norm(self, pred_semseg, semseg_label):
        if pred_semseg.shape[-1] != semseg_label.shape[-1]:
            pred_semseg=nn.functional.interpolate(pred_semseg,size=semseg_label.shape[1:],mode='bilinear',align_corners=False)
        pred_semseg = F.log_softmax(pred_semseg, dim=1)
        loss = self.crit(pred_semseg, semseg_label)
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
        semseg_label[prob_pos>cfg.SEM.OHEM_POS] = 255
        pred_semseg = F.log_softmax(pred_semseg, dim=1)
        loss = self.crit(pred_semseg, semseg_label)
        acc = self.pixel_acc(pred_semseg, semseg_label)
        del scan_pos 
        del scan
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
            return_dict['metrics']['accuracy_pixel'] = acc
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

