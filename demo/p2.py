import sys
sys.path.append('./lib')
import pickle, os
import numpy as np
from PIL import Image
from torch import optim, nn
import torch
from torch.autograd import Variable
import nn as mynn
import cv2
from numpy import uint8
from modeling.model_builder_segdisp import Generalized_SEGDISP
from modeling.model_builder_semseg import Generalized_SEMSEG
from datasets.cityscapes_api import labels
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch_semseg
from datasets.roidb import combined_roidb_for_training, combined_roidb_for_training_semseg
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from torchvision import transforms

def dataloader(bs, gpus):
    roidb, ratio_list, ratio_index = \
            combined_roidb_for_training_semseg('cityscapes_semseg_val')
    sampler = MinibatchSampler(ratio_list, ratio_index)
    dataset = RoiDataLoader(
        roidb,
        19,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler,
        num_workers=gpus,
        collate_fn=collate_minibatch_semseg)
    return dataloader

"""
one_list []
inputs PIL image (h,w,c)
"""
def transfer_img(image, imgname, scale): #将图片切分成多块
    resize_h = int(np.round(scale * image_shape[0]))
    resize_w = int(np.round(scale * image_shape[1]))
    # image = np.array(image.resize((resize_w, resize_h)))
    image = PIL_to_tensor(image.resize((resize_w, resize_h))) # (1,3,720,720)
    im_shape = image.shape[-2:] #720 720
    print(im_shape)
    step_h = int(np.ceil(1.0 * im_shape[0] / input_size[0]))
    step_w = int(np.ceil(1.0 * im_shape[1] / input_size[1]))
    one_list = []
    for ih in range(step_h):
        for iw in range(step_w):
            input_small = torch.zeros([1,3] + input_size)
            max_h = min(im_shape[0], (ih + 1) * input_size[0])
            max_w = min(im_shape[1], (iw + 1) * input_size[1])
            in_h = max_h - ih * input_size[0]
            in_w = max_w - iw * input_size[1]
            input_small[:, :, 0: in_h, 0: in_w] = image[:, :, ih * input_size[0]: max_h,
                    iw * input_size[1]: max_w]
            print(input_small.shape)
            img_small = tensor_to_PIL(input_small)
            one_list.append(img_small)

    return one_list, imgname.split('/')[-1], [step_h, step_w]


"""
将tensor数据转为imgs
ts (b, c, h, w )
img uint8(h, w, c)
"""
def tensor_to_images(ts): #
    print(ts.shape)
    imgs = []   # batch size个img
    for i in range(ts.shape[0]):
        tensor_img = ts[i]  # chw
        # tensor_data += torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1)  # 加均值 为了显示才需要
        tensor_img = tensor_img.permute(1, 2, 0)  # (720,720,3) hwc
        np_data = tensor_img.numpy().astype('uint8')
        img = Image.fromarray(np_data).convert('RGB')
        imgs.append(img)
        img.save('img_1.png')
    return imgs

"""
image (h,w,c)
tensorData Tensor(1, c, h, w)  
"""
def image_to_tensorData(img):
    np_data= np.array(img)
    tensor_data = torch.from_numpy(np_data)
    tensor_data = tensor_data.permute(2, 0, 1)
    tensor_data = tensor_data.unsqueeze(0)
    return tensor_data

def npdata_to_img(np_data):
    return Image.fromarray(np_data).convert('RGB')

def image_to_npdata(img):
    return np.array(img)

def is_equal(l1, l2, keys):
    len1 = len(l1)
    len2 = len(l2)
    if len1 != len2:
        return False

    for i in range(len1):
        if not torch.equal(l1[i], l2[i]):
            return False
        print(keys[i], 'equal')

    print('all equal')
    return True


def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

def tensor_to_PIL(tensor):
    image = tensor
    image = image.squeeze(0)
    image = unloader(image)
    return image


cfg_file = 'configs/baselines/e2e_pspnet-101_2x.yaml'
cfg_from_file(cfg_file)

print(cfg.DATA_DIR)










