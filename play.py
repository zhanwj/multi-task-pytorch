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

def transfer_tensordata(tensordata, imgname): #将Tensor图片切分成多块
    im_shape = tensordata.shape[-2:] #720 720
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
            input_small[:, :, 0: in_h, 0: in_w] = tensordata[:, :, ih * input_size[0]: max_h,
                    iw * input_size[1]: max_w]
            print(input_small.shape)
            one_list.append(input_small)

    return one_list, imgname.split('/')[-1], [step_h, step_w]


"""
imgs [pred_small_npdata]
return img npdata
"""
def merge_small_img(pred_list, index, scale=1):
    step_h, step_w = index
    for ih in range(step_h):
        for iw in range(step_w):
            if iw == 0:
                pred_w = pred_list[ih * step_w]
            else:
                pred_w = np.concatenate((pred_w, pred_list[ih * step_w + iw]), axis=1)
        if ih == 0:
            pred_prob = pred_w
        else:
            pred_prob = np.concatenate((pred_prob, pred_w), axis=0)
    max_h = int(np.round(image_shape[0] * scale))
    max_w = int(np.round(image_shape[1] * scale))
    pred_prob = pred_prob[0: max_h, 0: max_w, :]  # h,w,c
    return pred_prob


"""
将tensor数据转为imgs
ts (b, c, h, w )
img uint8(h, w, c)
"""
def tensor_to_images(ts): #
    ts = ts.squeeze(0)
    ts += torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1)  # 加均值 为了显示才需要
    tensor_img = ts.permute(1, 2, 0)  # (720,720,3) hwc
    np_data = tensor_img.numpy().astype('uint8')
    img = Image.fromarray(np_data).convert('RGB')
    return img

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

devices_ids=[5]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(ids) for ids in devices_ids])

pretrained_model = './output/pspnet50_2gpu_single_scale/Oct20-12-41-16_localhost.localdomain/ckpt/model_17_1486.pth'
checkpoint = torch.load(pretrained_model)

net = Generalized_SEMSEG()
net.load_state_dict(checkpoint['model'])
net = mynn.DataParallel(net.to('cuda'), minibatch=True)
net.eval()

# params = net.state_dict() #查看权重是否导入
# params_ckpt = checkpoint['model']
# a = list(params.values())
# b = list(params_ckpt.values())
# keys = list(params.keys())
# print(is_equal(a, b, keys))

len_gpus = len(devices_ids)
batch_size = 1 * len_gpus
dataloader= dataloader(batch_size, len_gpus)

loader = transforms.Compose([
    transforms.ToTensor(),
])

unloader = transforms.ToPILImage()

transColor = {label.trainId: label.color for label in labels}

for i, input_data in zip(range(10), dataloader):
    print(input_data.keys())
    # print(input_data['data'][0].shape) #torch.Size([1, 3, 720, 720])
    # print(type(input_data['data'][0])) #<class 'torch.Tensor'>

    # datas = input_data['data']
    # for tdata in datas:
    #     img = tensor_to_PIL(tdata)
    #     img.save('./seg_pred_pic/test' + str(i) + '.png')

    # for j, tesor_data in datas:
    # tensor_data= datas[0]
    # tensor_data = tensor_data.squeeze(0)
    # print(tensor_data.shape)
    # tensor_data = tensor_data.permute(1, 2, 0)
    # print(tensor_data.shape)
    # image = unloader(tensor_data)
    # print(type(image))
    # print(image)
    # print(image.size)
    # image.save('./seg_pred_pic/test' + str(i) + '.png')

    # for key in input_data:
    #     if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
    #         input_data[key] = list(map(lambda x: Variable(x).to('cuda'), input_data[key]))


    image_shape = [720, 720]
    input_size = [360, 360]
    aug_scale = [1]

    # tensor_data = input_data['data'][0]
    # imgs = tensor_to_images(tensor_data)  # 返回batch size个img
    for ii, img_tensor in enumerate(input_data['data']):
        for scale_i, scale in enumerate(aug_scale):
            img = tensor_to_PIL(img_tensor)
            img.save('./seg_pred_pic/ori'+ str(i) +'.png')

            img_add_mean = tensor_to_images(img_tensor)
            img_add_mean.save('./seg_pred_pic/ori_add_mean'+ str(i) +'.png')

            out = net.forward([Variable(img_tensor).cuda()])
            out = out['pred_semseg'][0]
            cfg_size = list(np.array(cfg.SEM.INPUT_SIZE))
            out = nn.functional.interpolate(  # 上采样
                out, size=cfg_size,
                mode='bilinear', align_corners=False)
            pred = out.max(1)[1].squeeze().cpu().data.numpy()
            assert np.all(pred >= 0) and np.all(pred < 19), 'error in prediction'
            colors = np.zeros(list(pred.shape) + [3], dtype='uint8')  # 19 1024 2048 3

            for ic in range(pred.shape[0]):
                for ir in range(pred.shape[1]):
                    colors[ic, ir, :] = transColor[pred[ic, ir]]
            img = Image.fromarray(colors)
            img.save('./seg_pred_pic/mypred' + str(i) + '.png')

            """
            切分图片
            """
            # one_list, image_name, index = transfer_img(img, imgname=str(i) + 'bs_' + str(ii), scale=scale)
            one_list, image_name, index = transfer_tensordata(img_tensor, imgname=str(i) + 'bs_' + str(ii))
            print(len(one_list))
            pred_list = []
            for ism, img_small_tensor in enumerate(one_list):
                img_small = tensor_to_PIL(img_small_tensor)
                img_small.save('./seg_pred_pic/ori'+ str(i) +'_small' + str(ism) +'.png')
                print('./seg_pred_pic/ori'+ str(i) +'_small' + str(ism) +'.png')


                out = net.forward([Variable(img_small_tensor).cuda()])
                out = out['pred_semseg'][0]
                cfg_size = list(np.array(cfg.SEM.INPUT_SIZE) // index[0])
                out = nn.functional.interpolate(  # 上采样
                    out, size=cfg_size,
                    mode='bilinear', align_corners=False)
                pred = out.max(1)[1].squeeze().cpu().data.numpy()
                assert np.all(pred >= 0) and np.all(pred < 19), 'error in prediction'
                colors = np.zeros(list(pred.shape) + [3], dtype='uint8')

                for ic in range(pred.shape[0]):
                    for ir in range(pred.shape[1]):
                        colors[ic, ir, :] = transColor[pred[ic, ir]]
                img_small = Image.fromarray(colors)
                img_small.save('./seg_pred_pic/mypred' + str(i) + '_small_img' + str(ism) +'.png')

                pred_list.append(colors)

            mereg_npdata = merge_small_img(pred_list, index)
            merge_img = Image.fromarray(mereg_npdata)
            merge_img.save('./seg_pred_pic/mypred_merge' + str(i) + '.png')

    # break
    # tensor_to_PIL(input_data['data'][0]).save('./seg_pred_pic/imput.png')
    # out = net.forward([input_data['data'][0]])
    # # print(out['pred_semseg'][0].shape)
    # out = out['pred_semseg'][0]
    # out = nn.functional.interpolate( #上采样
    #     out, size=cfg.SEM.INPUT_SIZE,
    #     mode='bilinear', align_corners=False)
    #
    # pred = out.max(1)[1].squeeze().cpu().data.numpy()
    #
    # assert np.all(pred >= 0) and np.all(pred < 19), 'error in prediction'
    # colors = np.zeros(list(pred.shape) + [3], dtype='uint8')  # 19 1024 2048 3
    #
    # transColor = {label.trainId: label.color for label in labels}
    # for ic in range(pred.shape[0]):
    #     for ir in range(pred.shape[1]):
    #         colors[ic, ir, :] = transColor[pred[ic, ir]]
    #
    # print(colors.shape)
    #
    # img = Image.fromarray(colors)
    # img.save('./seg_pred_pic/mypred'+ str(i) +'.png')

    # loss = net(**input_data)
    # print(loss)









