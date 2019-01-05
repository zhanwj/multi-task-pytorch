#encoding=utf-8
from tensorboardX import SummaryWriter
import argparse
import distutils.util
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict
import pynvml
import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
import scipy.misc as scmi
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

#import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training, combined_roidb_for_training_semseg
from modeling.model_builder import Generalized_RCNN
from modeling.model_builder_3DSD import Generalized_3DSD
from modeling.model_builder_segdisp import Generalized_SEGDISP
from modeling.model_builder_semseg_bat import Generalized_SEMSEG
from modeling.model_builder_segcspn import Generalized_SEGCSPN
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch_semseg
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats
#图片处理
#import sys
#sys.path.append('./lib')
##import optIO
#import numpy as np
import time
#import torch
#import yaml
#import os
#import scipy.io as scio
#import scipy.misc as smi
#import cv2
#import logging as logger
#import pickle
#from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
#from modeling.model_builder_semseg import Generalized_SEMSEG
#from modeling.model_builder_segdisp_original import Generalized_SEGDISP
## from modeling.model_builder_segdisp import Generalized_SEGDISP
from datasets.cityscapes_api import labels
#from torch.autograd import Variable
#from torch import nn
from PIL import Image
#logger.basicConfig(format='Process:',level=logger.DEBUG)

def write(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
def load(filename):
    with open(filename, 'rb') as fp:
        data=pickle.load(fp)
    return data

def argument():
    import argparse
    parser = argparse.ArgumentParser(description='Process prediction.')
    parser.add_argument('--dataset', dest='dataset', help='give a dataset name', default='cityscapes', type=str)
    # parser.add_argument('--config', dest='config', help='which file to restore', default='configs/baselines/e2e_pspnet-101_2x.yaml', type=str)
    parser.add_argument('--config', dest='config', help='which file to restore', default='./configs/baselines/e2e_ubernet-101_2x.yaml', type=str)
    parser.add_argument('--save_file', dest='save_file', help='where to save file', default='./seg_pred_pic/pred_sem_val_500_ubernet50_plateau/', type=str)
    parser.add_argument('--gpu', dest='gpu', help='give a gpu to train network', default=0, type=int)
    parser.add_argument('--input_size', dest='input_size', help='input size of network', nargs='+', default=[720,720], type=int)
    parser.add_argument('--aug_scale', dest='aug_scale', help='scale image of network', nargs='+', default=[1440], type=int)
    parser.add_argument('--network', dest='network', help='network name', default='Generalized_SEMSEG', type=str)
    # parser.add_argument('--network', dest='network', help='network name', default='Generalized_SEMSEG', type=str)
    parser.add_argument('--pretrained_model', dest='premodel', help='path to pretrained model', default='./output/ubernet50_multiscale_ReduceLROnPlateau_40epochs_720/e2e_ubernet-101_2x/Dec07-10-56-09_localhost.localdomain/ckpt//model_59_1486.pth', type=str)
    parser.add_argument('--prefix_semseg', dest='prefix_semseg', help='output name of network', default='pred_semseg', type=str)
    parser.add_argument('--prefix_disp', dest='prefix_disp', help='output name of network', default='pred_disp', type=str)
    parser.add_argument('--prefix_average', dest='prefix_average', help='output name of network', default='pred_deepsup', type=str)
    parser.add_argument('--merge_method', dest='merge_method', help='merge method for MS', default='ave', type=str)
    parser.add_argument('--index_start', dest='index_start', help='predict from index_start', default=0, type=int)
    parser.add_argument('--index_end', dest='index_end', help='predict end with index_end', default=500, type=int)
    parser.add_argument('--save_final_prob', dest='save_final_prob', help='to save prob for each class',  default=0, type=int)
    args = parser.parse_args()
    return args

def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

class TestNet(object):
    def __init__(self, args):
        # assert False, 'merge config'
        cfg_from_file(args.config)
        cfg.TRAIN.IMS_PER_BATCH = 1
        args.aug_scale=cfg.TRAIN.SCALES
        args.input_size=cfg.SEM.INPUT_SIZE
        self.input_size=cfg.SEM.INPUT_SIZE
        print ('test scale:',args.aug_scale)
        self._cur = 0
        if args.network == 'Generalized_SEGDISP':
            self.load_image = self.load_segdisp_image
            to_test = to_test_segdisp
        elif args.network=='Generalized_SEMSEG':
            self.load_image = self.load_semseg_image
            to_test = to_test_semseg
            
        else:
            self.load_image = self.load_semseg_image
            to_test = to_test_semseg(args)
        if 'cityscape' in args.dataset:
            self.input_size = args.input_size
            self.aug_scale = args.aug_scale
            if 'train_on_val' in args.dataset:
                self.label_root = os.path.join(os.getcwd(),'lib/datasets/data/cityscapes/annotations/Cityscape_disp_SegFlow_train.txt')
            elif 'train' in args.dataset :
                self.label_root = os.path.join(os.getcwd(),'lib/datasets/data/cityscapes/annotations/train.txt')
            else:
                self.label_root = os.path.join(os.getcwd(),'lib/datasets/data/cityscapes/annotations/val.txt')
            # self.label_root = os.path.join(os.getcwd(),'lib/datasets/data/cityscapes/label_info_fine/test.txt')
            #self.label_root = os.path.join(os.getcwd(),'citycapes/label_info/onlytrain_label_citycapes_right.txt')
            self.num_class = 19
            self.image_shape=[1024, 2048]
        self.load_listname(args)
        self.pretrained_model=args.premodel
        #transformer label
        self.transLabel = {label.trainId : label.id for label in labels} ##
        self.transColor = {label.trainId : label.color for label in labels}

    def up_sample(self, data):
        return nn.functional.interpolate(
            data, size = self.image_shape, mode='bilinear', align_corners=False)

    def load_listname(self, args):
        indexlist = [line.split() for line in open(self.label_root,'r').read().splitlines()]
        self.indexlist = indexlist[args.index_start: args.index_end]

    def load_net(self, args):
        #set device of gpu
        # assert False, 'use os.env to set device gpu %s'%str(args.gpus)
        #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(ids) for ids in args.gpu])
        print (cfg.MODEL.TYPE)
        self.net = eval(cfg.MODEL.TYPE)()
        #init weight
        # pretrained_model = args.premodel
        
        print("loading pspnet weights")
        state_dict={}
        pretrained=torch.load(self.pretrained_model, map_location=lambda storage, loc: storage)
        pretrained = pretrained['model']
        self.net.load_state_dict(pretrained,strict=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        self.net.encoder.to('cuda')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu+1)
        self.net.decoder.to('cuda')
        print("weights load success")
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        del pretrained

        #checkpoint = torch.load(self.pretrained_model)
        #self.net.load_state_dict(checkpoint['model'])
        #self.net.to('cuda')
        #self.net.eval()

    def load_segdisp_image(self, args):
        img_L, img_R, _, _ = self.indexlist[self._cur]
        image_L = cv2.imread(os.path.join(cfg.DATA_DIR, args.dataset, img_L))
        image_R = cv2.imread(os.path.join(cfg.DATA_DIR, args.dataset, img_R))
        cv2.imwrite(os.path.join(args.save_file, img_L.split('/')[-1]), image_L) #保存原图
        image = [image_L, image_R]
        self._cur += 1
        return image, img_L
        
    def load_semseg_image(self, args):
        imgname =  self.indexlist[self._cur][0].split('citycapes/')[-1]
        image = cv2.imread(os.path.join(cfg.DATA_DIR, args.dataset.split('_')[0], imgname))
        #cv2.imwrite(os.path.join(args.save_file, imgname.split('/')[-1]), image) #保存原图
        self._cur += 1
        return image, imgname

    # put image left and right into it, sparetion
    def transfer_img(self, image, imgname, scale, args): #将图片切分成多块
        assert np.all(args.input_size==cfg.SEM.INPUT_SIZE), 'cfg size must be same to args'
        resize_h = scale // 2
        resize_w = scale
        image = np.array(cv2.resize(image.copy(), (resize_w, resize_h)))
        crop_h_max = max(resize_h - args.input_size[0], 0)
        crop_w_max = max(resize_w - args.input_size[1], 0)
        step_h = 1 + int(np.ceil(1.0*crop_h_max/ args.input_size[0]))
        step_w = 1 + int(np.ceil(1.0*crop_w_max / args.input_size[1]))
        one_list = []
        tmp_name = os.path.join(args.save_file, imgname.split('/')[-1])
        boxes = []
        for ih in range(step_h):
            for iw in range(step_w):
                inputs = np.zeros(args.input_size+[3])
                crop_sh = min(ih*args.input_size[0], crop_h_max)
                crop_sw = min(iw*args.input_size[1], crop_w_max)
                crop_eh = min(resize_h, crop_sh+args.input_size[0])
                crop_ew = min(resize_w, crop_sw+args.input_size[1])
                in_eh = crop_eh - crop_sh
                in_ew = crop_ew - crop_sw
                inputs[0: in_eh, 0: in_ew] = image[crop_sh:crop_eh, crop_sw:crop_ew]
                #cv2.imwrite(tmp_name.replace('.png','%d_%d.png' %(ih, iw)), inputs)
                inputs = inputs[:, :, ::-1]
                inputs -= cfg.PIXEL_MEANS
                inputs = inputs.transpose(2, 0, 1)
                one_list.append(inputs.copy()) 
                boxes.append([crop_sh, crop_eh, crop_sw, crop_ew,in_eh,in_ew])
        #cv2.imwrite(tmp_name+'_.png', pred_prob[:1024, :2048, :])
        return one_list, imgname.split('/')[-1], boxes

    def save_pred(self, pred_list, image_name, scale_info, index, args): #拼起来
        assert np.all(args.input_size==cfg.SEM.INPUT_SIZE), 'cfg size must be same to args'
        assert np.all(list(pred_list[0].shape[2:]) == args.input_size), 'pred size is not same to input size'
        assert np.all(pred_list[0] >=0), 'pred must be output of softmax'
        assert pred_list[0].shape[0] == 1, 'only support one sample'
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        scale, scale_i = scale_info
        pred_prob=np.zeros(([1, self.num_class]+[scale//2, scale]))
        pred_smooth= np.zeros(([1, self.num_class]+[scale//2, scale]))
        for ids, ibox in enumerate(index):
            sh,eh,sw,ew,in_h,in_w=ibox
            pred_prob[:, :, sh:eh, sw:ew] += pred_list[ids][:, :, 0:in_h, 0:in_w]
            pred_smooth[:, :, sh:eh, sw:ew] += 1
        assert np.all(pred_smooth >=1), 'error merge'
        pred_prob /= pred_smooth
        return pred_prob
        #write( tmp_name+'_'+str(scale_i)+'_prob.pkl', pred_prob)

    def save_multi_results(self, pred_scales_list, image_name, args): #多尺寸
        num_scale = len(self.aug_scale)
        pred_prob = np.zeros([self.num_class]+self.image_shape)#(19, 1024, 2048)
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        for i, scale in enumerate(self.aug_scale):
            #scale_pred = load(tmp_name+'_'+str(i)+'_prob.pkl')#(1, 19, 90, 180)
            scale_pred = pred_scales_list[i] #(1, 19, 90, 180)
            assert scale_pred.shape[0]==1, 'only support one sample'
            #pred_prob += self.up_sample(torch.from_numpy(scale_pred))[0].numpy()
            if image_name.endswith('_disp'):
                pred_disp = cv2.resize(scale_pred[0][0], tuple(self.image_shape[::-1]), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(args.save_file,image_name.replace('.png','')) + '.png', pred_disp)
            else:
                for ic in range(self.num_class): ##
                    if args.merge_method == 'ave':
                        pred_prob[ic] += cv2.resize(scale_pred[0][ic], tuple(self.image_shape[::-1]), interpolation=cv2.INTER_LINEAR)
                    else:
                        pred_prob_i=cv2.resize(scale_pred[0][ic], tuple(self.image_shape[::-1]), interpolation=cv2.INTER_LINEAR)
                        pred_prob[ic] =np.max([pred_prob[ic], pred_prob_i], axis=0)
            #os.remove(tmp_name+'_'+str(i)+'_prob.pkl')
        if  args.save_final_prob:
            if args.merge_method == 'ave':
                pred_prob /= num_scale
            assert np.all(pred_prob <= 1), 'error scale'
            write(tmp_name+'_prob.pkl', pred_prob.astype(np.float16))
        else:
            pred_prob = np.argmax(pred_prob, axis=0) #(1024, 2048)
            self.transfer_label(image_name, pred_prob)

    def transfer_label(self, image_name, pred_prob): #图片可视化： 灰度图-》彩色图
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        assert np.all(pred_prob >=0) and np.all(pred_prob<19), 'error in prediction'
        colors = np.zeros(list(pred_prob.shape)+ [3], dtype='uint8') # 19 1024 2048 3
        for ic in range(pred_prob.shape[0]):
            for ir in range(pred_prob.shape[1]):
                colors[ic, ir, :] = self.transColor[ pred_prob[ic, ir] ]
        Idsave = Image.fromarray(colors.astype('uint8'))
        Idsave.save(tmp_name+'_color.png')

        pred_map = np.zeros_like(pred_prob)
        for k, v in self.transLabel.items():
            pred_map[pred_prob==k] = v
        cv2.imwrite(tmp_name+'_labelId.png', pred_map)

def to_test_segdisp(args):
    test_net = TestNet(args)
    test_net.load_net(args)
    shapes = [1, 3] + args.input_size
    for i in range(args.index_start, args.index_end):
        time_now = time.time()
        image, image_name = test_net.load_image(args) # 2 <class 'list'>
        for scale_i, scale in enumerate(test_net.aug_scale): #每种scale
            pred_list = [] ##预测的数据
            pred_list_disp = []
            pred_deepsup_list_disp = []
            one_list_L, image_name, index = test_net.transfer_img(image[0], image_name, scale, args)
            one_list_R, image_name, index = test_net.transfer_img(image[1], image_name, scale, args)
            for isave, (im_L, im_R) in enumerate(zip(one_list_L, one_list_R)): #剪成多张图片 一张一张喂进去
                im = np.concatenate((im_L.reshape(shapes), im_R.reshape(shapes)), axis=0) #(2, 3, 720, 720)
                input_data = Variable(torch.from_numpy(im).float(), requires_grad=False).cuda() #torch.Size([2, 3, 720, 720])
                pred_dict = test_net.net(input_data)
                pred_list.append(pred_dict[args.prefix_semseg].detach().cpu().numpy())
                pred_list_disp.append(pred_dict[args.prefix_disp].detach().cpu().numpy())
                pred_deepsup_list_disp.append(pred_dict['pred_disp_deepsup'].detach().cpu().numpy())
            test_net.save_pred(pred_list, image_name, [scale, scale_i], index, args) #之后将图片合起来
            test_net.save_pred(pred_list_disp, image_name + '_disp', [scale, scale_i], index, args)
            test_net.save_pred(pred_deepsup_list_disp, image_name + '_average_disp', [scale, scale_i], index, args)
        test_net.save_multi_results(image_name,  args)
        test_net.save_multi_results(image_name+'_disp',  args)
        test_net.save_multi_results(image_name + '_average_disp', args)
        cost_time = time.time() - time_now
        print ('{} cost {}, remain time:{}'.format(image_name, cost_time, (args.index_end - i - 1)*cost_time))

def to_test_semseg(args):
    test_net = TestNet(args)
    test_net.load_net(args)
    net_stride = 8 if '8' in cfg.SEM.ARCH_ENCODER else 16
    for i in range(args.index_start, args.index_end):
        time_now = time.time()
        image, image_name = test_net.load_image(args)
        pred_final_list = []
        for scale_i, scale in enumerate(test_net.aug_scale): #每种scale
            scale = round2nearest_multiple(scale, net_stride)
            cfg.SEM.INPUT_SIZE=[scale//2, scale] if scale <= 5000 else [scale//4, scale//2]
            args.input_size = cfg.SEM.INPUT_SIZE
            #cfg_from_file(args.config)
            pred_list = [] ##预测的数据
            #pred_deepsup_list = []
            one_list, image_name, index = test_net.transfer_img(image, image_name, scale, args)
            for isave, im in enumerate(one_list): #剪成c多张图片 一张一张喂进去
                for iflip in range(1):
                    if iflip == 0:
                        input_data = Variable(torch.from_numpy(im[np.newaxis,:]).float(), requires_grad=False).cuda()
                        #pred_dict = test_net.net(input_data)[args.prefix_semseg].detach().cpu().numpy()
                        features = test_net.net.encoder(input_data, return_feature_maps=True)
                        pred_dict = test_net.net.decoder(features, segSize=cfg.SEM.INPUT_SIZE)[args.prefix_semseg].detach().cpu().numpy()
                    else:
                        im_flip = im[:, :, ::-1].copy()
                        input_data = Variable(torch.from_numpy(im_flip[np.newaxis,:]).float(), requires_grad=False).cuda()
                        #pred_dict += test_net.net(input_data)[args.prefix_semseg].detach().cpu().numpy()[:, :, :, ::-1]
                        features = test_net.net.encoder(input_data, return_feature_maps=True)
                        pred_dict += test_net.net.decoder(features, segSize=cfg.SEM.INPUT_SIZE)[args.prefix_semseg].detach().cpu().numpy()
                pred_list.append(pred_dict/(1+iflip))
            pred_final_list.append(test_net.save_pred(pred_list, image_name, [scale, scale_i], index, args))
            del pred_list
            del one_list
            #test_net.save_pred(pred_deepsup_list, image_name + '_average', [scale, scale_i], index, args)
        test_net.save_multi_results(pred_final_list, image_name,  args)
        del pred_final_list
        #test_net.save_multi_results(image_name + '_average',  args)
        cost_time = time.time() - time_now
        print ('{} cost {}s , remain time:{}s '.format(image_name, cost_time, (args.index_end - i - 1)*cost_time))


if __name__ == '__main__':
    args = argument()
    if not os.path.exists(args.save_file):
        os.makedirs(args.save_file)
        os.makedirs(args.save_file+'_labelId')
    if 'SEGDISP' in args.network:
        to_test_segdisp(args)
    else:
        to_test_semseg(args)
