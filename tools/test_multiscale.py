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
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training, combined_roidb_for_training_semseg
from modeling.model_builder import Generalized_RCNN
from modeling.model_builder_3DSD import Generalized_3DSD
from modeling.model_builder_segdisp import Generalized_SEGDISP
from modeling.model_builder_semseg_bat import Generalized_SEMSEG
from modeling.model_builder_psp_pretrained_test import Generalized_SEMSEG
from modeling.model_builder_segcspn import Generalized_SEGCSPN
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch_semseg
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats
#图片处理
#import sys
##sys.path.append('./lib')
##import optIO
#import numpy as np
#import time
#import torch
#import os
#import yaml
#import scipy.io as scio
#import scipy.misc as smi
#import cv2
#import logging as logger
#import pickle
#import _init_paths
#from modeling.model_builder_semseg import Generalized_SEMSEG
#from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
#from modeling.model_builder_semseg import Generalized_SEMSEG
#from modeling.model_builder_segdisp_original import Generalized_SEGDISP
## from modeling.model_builder_segdisp import Generalized_SEGDISP
#from datasets.cityscapes_api import labels
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
    parser.add_argument('--config', dest='config', help='which file to restore', default='../configs/baselinese/e2e_psp_pretrained_test.yaml', type=str)
    parser.add_argument('--save_file', dest='save_file', help='where to save file', default='./seg_pred_pic/pred_segdisp_val_500_', type=str)
    parser.add_argument('--gpu', dest='gpu', help='give a gpu to train network', default=[2], type=list)
    parser.add_argument('--input_size', dest='input_size', help='input size of network', nargs='+', default=[512,1024], type=int)
    parser.add_argument('--aug_scale', dest='aug_scale', help='scale image of network', nargs='+', default=[1024], type=int)
    parser.add_argument('--network', dest='network', help='network name', default='Generalized_SEMSEG', type=str)
    # parser.add_argument('--network', dest='network', help='network name', default='Generalized_SEMSEG', type=str)
    parser.add_argument('--pretrained_model', dest='premodel', help='path to pretrained model', default='../output/pspnet_poly_without_mulitScale/e2e_psp_pretrained_test/Dec05-03-35-50_localhost.localdomain/ckpt/model_39_178.pth', type=str)
    parser.add_argument('--prefix_semseg', dest='prefix_semseg', help='output name of network', default='pred_semseg', type=str)
    parser.add_argument('--prefix_disp', dest='prefix_disp', help='output name of network', default='pred_disp', type=str)
    parser.add_argument('--prefix_average', dest='prefix_average', help='output name of network', default='pred_semseg_average', type=str)
    parser.add_argument('--index_start', dest='index_start', help='predict from index_start', default=0, type=int)
    parser.add_argument('--index_end', dest='index_end', help='predict end with index_end', default=500, type=int)
    args = parser.parse_args()
    return args


class TestNet(object):
    def __init__(self, args):
        # assert False, 'merge config'
        cfg_from_file(args.config)
        cfg.TRAIN.IMS_PER_BATCH = 1
        self._cur = 0
        if 'SEGDISP' in args.network:
            self.load_image = self.load_segdisp_image
            to_test = to_test_segdisp
        else:
            self.load_image = self.load_semseg_image
            to_test = to_test_semseg()
        if args.dataset == 'cityscapes':
            self.input_size = args.input_size
            self.aug_scale = args.aug_scale
            self.label_root = os.path.join(os.getcwd(),'lib/datasets/data/cityscapes/annotations/val.txt')
            # self.label_root = os.path.join(os.getcwd(),'lib/datasets/data/cityscapes/label_info_fine/test.txt')
            #self.label_root = os.path.join(os.getcwd(),'citycapes/label_info/onlytrain_label_citycapes_right.txt')
            self.num_class = 19
            self.image_shape=[1024, 2048]
        self.load_listname(args)

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
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(ids) for ids in args.gpu])
        self.net = eval(args.network)()
        #init weight
        # pretrained_model = args.premodel
        checkpoint = torch.load(self.pretrained_model)
        self.net.load_state_dict(checkpoint['model'])
        self.net.to('cuda')
        self.net.eval()

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
        image = cv2.imread(os.path.join(cfg.DATA_DIR, args.dataset, imgname))
        cv2.imwrite(os.path.join(args.save_file, imgname.split('/')[-1]), image) #保存原图
        self._cur += 1
        return image, imgname

    # put image left and right into it, sparetion
    def transfer_img(self, image, imgname, scale, args): #将图片切分成多块
        #image = cv2.imread(imgname)
        #image = cv2.resize(image, None,None,fx=scale,fy=scale, interpolation=cv2.INTER_NEAREST)
        resize_h = scale // 2
        resize_w = scale
        cv2.resize(image, (resize_w, resize_h))
        image = np.array(cv2.resize(image, (resize_w, resize_h)))
        im_shape = image.shape[:2] #720 1440
        step_h = int(np.ceil(1.0*im_shape[0] / args.input_size[0]))
        step_w = int(np.ceil(1.0*im_shape[1] / args.input_size[1]))
        one_list = []
        tmp_name = os.path.join(args.save_file, imgname.split('/')[-1])
        for ih in range(step_h):
            for iw in range(step_w):
                inputs = np.zeros(args.input_size+[3])
                max_h = min(im_shape[0], (ih+1)*args.input_size[0])
                max_w = min(im_shape[1], (iw+1)*args.input_size[1])
                in_h= max_h - ih*args.input_size[0]
                in_w= max_w - iw*args.input_size[1]
                inputs[0: in_h, 0: in_w] = image[ih*args.input_size[0]: max_h,
                        iw*args.input_size[1]: max_w]
                #cv2.imwrite(tmp_name.replace('.png','%d_%d.png' %(ih, iw)), inputs)
                inputs = inputs[:, :, ::-1]
                inputs -= cfg.PIXEL_MEANS
                inputs = inputs.transpose(2, 0, 1)
                one_list.append(inputs.copy()) #因为copy操作可以在原先的numpy变量中创造一个新的不适用负索引的numpy变量。

        #cv2.imwrite(tmp_name+'_.png', pred_prob[:1024, :2048, :])
        return one_list, imgname.split('/')[-1], [step_h, step_w]

    def save_pred(self, pred_list, image_name, scale_info, index, args): #拼起来
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        step_h, step_w = index
        scale, scale_i = scale_info
        for ih in range(step_h):
            for iw in range(step_w):
                if iw == 0:
                    pred_w = pred_list[ih*step_w]
                else:
                    pred_w = np.concatenate((pred_w, pred_list[ih*step_w+iw]), axis=3)
            if ih == 0:
                pred_prob = pred_w
            else:
                pred_prob = np.concatenate((pred_prob, pred_w), axis=2)
        max_h = scale // 2
        max_w = scale
        pred_prob = pred_prob[:, 0: max_h, 0: max_w] ## c,h,w
        write( tmp_name+'_'+str(scale_i)+'_prob.pkl', pred_prob)

    def save_multi_results(self, image_name, args): #多尺寸
        num_scale = len(self.aug_scale)
        pred_prob = np.zeros([self.num_class]+self.image_shape)#(19, 1024, 2048)
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        for i, scale in enumerate(self.aug_scale):
            scale_pred = load(tmp_name+'_'+str(i)+'_prob.pkl')#(1, 19, 90, 180)
            # pred_prob += self.up_sample(torch.from_numpy(scale_pred))[0].numpy()
            if image_name.endswith('_disp'):
                pred_disp = cv2.resize(scale_pred[0][0], tuple(self.image_shape[::-1]), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(args.save_file,image_name.replace('.png','')) + '.png', pred_disp)
            else:
                for ic in range(self.num_class): ##
                    pred_prob[ic] += cv2.resize(scale_pred[0][ic], tuple(self.image_shape[::-1]), interpolation=cv2.INTER_LINEAR)
            os.remove(tmp_name+'_'+str(i)+'_prob.pkl')
        pred_prob = np.argmax(pred_prob, axis=0) #(1024, 2048)
        if image_name.endswith('_disp'):
            pass
        else:
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
    for i in range(args.index_start, args.index_end):
        time_now = time.time()
        image, image_name = test_net.load_image(args)
        for scale_i, scale in enumerate(test_net.aug_scale): #每种scale
            pred_list = [] ##预测的数据
            pred_deepsup_list = []
            one_list, image_name, index = test_net.transfer_img(image, image_name, scale, args)
            for isave, im in enumerate(one_list): #剪成多张图片 一张一张喂进去
                input_data = Variable(torch.from_numpy(im[np.newaxis,:]).float(), requires_grad=False).cuda()
                pred_dict = test_net.net(input_data)
                pred_list.append((pred_dict[args.prefix_semseg]).detach().cpu().numpy())
                pred_deepsup_list.append((pred_dict[args.prefix_average]).detach().cpu().numpy())
            test_net.save_pred(pred_list, image_name, [scale, scale_i], index, args) #之后将图片合起来
            test_net.save_pred(pred_deepsup_list, image_name + '_average', [scale, scale_i], index, args)
        test_net.save_multi_results(image_name,  args)
        test_net.save_multi_results(image_name + '_average',  args)
        cost_time = time.time() - time_now
        print ('{} cost {}, remain time:{}'.format(image_name, cost_time, (args.index_end - i - 1)*cost_time))


if __name__ == '__main__':
    args = argument()
    if 'SEGDISP' in args.network:
        to_test_segdisp(args)
    else:
        to_test_semseg(args)
