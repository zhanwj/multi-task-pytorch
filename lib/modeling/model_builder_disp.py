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

        self.crit = nn.NLLLoss(ignore_index=255)
        self.deep_sup_scale = cfg.SEM.DEEP_SUB_SCALE
        
    
        if cfg.DISP.ORIGINAL:
            self.conv_last_disp = nn.Conv2d(cfg.DISP.DIM*2,1,kernel_size=3,padding=1,bias=False)
            self.conv_last_disp.apply(self._init_weights_normal)

        if cfg.DISP.USE_CRL_DISPFUL:
            self.conv_last_disp = CRL.DispFulNetSubset()
        if cfg.DISP.USE_CRL_DISPRES:
            self.conv_last_disp = CRL.DispResNetSubset(),
            self.conv_last_disp.apply(self._init_weights_normal)

                
        self.SmoothL1Loss = nn.SmoothL1Loss(reduction='none') #elementwise_mean
        #self.disp_loss = self._disp_loss
        #self.hardtanh = nn.Hardtanh(-1, 1,inplace=True)
        if len(cfg.SEM.DOWNSAMPLE) > 1:
            self.disp_deepsup=nn.Sequential(
                nn.Conv2d(cfg.SEM.FC_DIM // 2, cfg.SEM.FC_DIM // 4, kernel_size=3, stride=1, padding=1, bias=None),
                SynchronizedBatchNorm2d(cfg.SEM.FC_DIM // 4),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(cfg.SEM.FC_DIM // 4, 1, kernel_size=3, stride=1, padding=1, bias=None)
            )
            self.disp_deepsup.apply(self._init_weights_kaiming)
        
        
    def disp_loss(self, pred, targe, sparse=True):
        assert len(targe.size()) == 3, 'expect targe size is b,h,w'
        b, channel, h, w = pred.size()
        loss = 0.
        targe = targe.unsqueeze(1)

        self.iter = 5
        if sparse:
            targe = F.adaptive_max_pool2d(targe, (h, w))
        pred=pred.view(b,-1).contiguous()
        targe=targe.view(b,-1).contiguous()
        EPE_map = self.SmoothL1Loss(pred, targe)
        #print("EPE_map mean:",EPE_map.mean())
        positive = (targe > 0).long()
        #ignore = (EPE_map*positive)
        #print("EPE_map mean:",EPE_map.mean())
        #EPE_map = EPE_map[positive]
        loss = EPE_map.mean()
        return loss





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
            if cfg.SEM.DECODER_TYPE.endswith('deepsup'): # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(data, return_feature_maps=True))
                
            if(cfg.DISP.USE_CRL_DISPFUL):
                resFeature = self.encoder(data,return_feature_maps=True)
                pred = self.decoder(resFeature)
            if(cfg.SEM.USE_RESNET):
                pred = self.decoder(self.encoder(data))
            if(cfg.DISP.ORIGINAL):
                pred = self.decoder(self.encoder(data, return_feature_maps=False))
                    
            
            step=2**cfg.SEM.DOWNSAMPLE[0]
            input_size=[cfg.SEM.INPUT_SIZE[0]//step, cfg.SEM.INPUT_SIZE[1]//step]


            pred_semseg, pred_disp = torch.split(pred, cfg.TRAIN.IMS_PER_BATCH, dim=0)
            
            if cfg.DISP.USE_CRL_DISPFUL:
                pred_disp = self.conv_last_disp(pred_disp,pred_semseg8X,left,resFeature)
            if cfg.DISP.USE_CRL_DISPRES:
                dataRes=[]
                pred_disp1X=nn.functional.interpolate(pred_disp,size=input_size,mode='bilinear',align_corners=False)
                dataRes.append(left)
                dataRes.append(right)
                dataRes.append(pred_disp1X)
                pred_disp = self.conv_last_disp(dataRes)
            if not (cfg.DISP.USE_CRL_DISPRES or cfg.DISP.USE_CRL_DISPFUL) :
                pred_disp = self.conv_last_disp(pred_disp)
                pred_disp=nn.functional.interpolate(pred_disp,size=input_size,mode='bilinear',align_corners=False)

            # not cfg.DISP.USE_CRL:
            
            
            #pred_disp=nn.functional.interpolate(pred_disp,scale_factor=8,mode='bilinear',align_corners=False)
            #cfg.SEM.INPUT_SIZE
            #print("label:",label_info['%s_0'%cfg.SEM.OUTPUT_PREFIX].shape)
            
            if cfg.DISP.ORIGINAL:     
                loss_disp = self.disp_loss(pred_disp, 
                    label_info['%s_0'%cfg.DISP.OUTPUT_PREFIX])
                EPE = loss_disp

            if cfg.DISP.USE_CRL_DISPFUL:
                if cfg.DISP.USE_MULTISCALELOSS:
                    loss_disp,EPE = self.multiScaleLoss(pred_disp,label_info['%s_0'%cfg.DISP.OUTPUT_PREFIX])
                else:
                    loss_disp = self.disp_loss(pred_disp[-1], 
                        label_info['%s_0'%cfg.DISP.OUTPUT_PREFIX])
                    EPE = loss_disp
            
            disp_image=''
                

            #res = torch.cat((pred_disp[0][0], label_info['%s_0'%cfg.DISP.OUTPUT_PREFIX][:,::step, ::step]),dim=2)
            #semseg_image= torch.argmax(pred_semseg[0],dim=0)
                
            semseg_image=torch.argmax(pred_semseg[0],dim=0)*10
            if cfg.DISP.USE_MULTISCALELOSS:
                res = torch.cat((pred_disp[0][0][0], label_info['%s_0'%cfg.DISP.OUTPUT_PREFIX][0,::step, ::step]),dim=1).cpu().detach().numpy()
            else:
                res = torch.cat((pred_disp[0][0], label_info['%s_0'%cfg.DISP.OUTPUT_PREFIX][0,::step, ::step]),dim=1).cpu().detach().numpy()
                
            cv2.imwrite('pred_disp_{}.png'.format(self.iter), res)
            disp_image=res
            

            return_dict['losses']['loss_disp'] = loss_disp
            return_dict['metrics']['EPE'] = EPE
            return_dict['disp_image'] = disp_image

            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
            #print("check")
        else: # inference
            pred = self.decoder(self.encoder(data, return_feature_maps=True), segSize=segSize)
            pred_semseg, pred_disp = torch.split(pred, 1, dim=0)
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
