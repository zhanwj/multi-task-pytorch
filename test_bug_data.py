import sys
sys.path.append('./lib')
import torch
import lib.modeling.resnet as resnet
import lib.modeling.semseg_heads as snet
import torch.nn as nn
import torch.optim as optim
import utils.resnet_weights_helper as resnet_utils
from torch.autograd import Variable
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch_semseg
from datasets.roidb import combined_roidb_for_training, combined_roidb_for_training_semseg
import os
import numpy as np
import nn as mynn
import cv2
from modeling.model_builder_segdisp import Generalized_SEGDISP
from modeling.model_builder_semseg import Generalized_SEMSEG
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
#load net
class load_net(nn.Module):
    def __init__(self):
        super(load_net, self).__init__()

        build=snet.ModelBuilder()
        fc_dim = 2048
        self.encoder = build.build_encoder(
                arch= 'resnet50_dilated8',
                fc_dim=fc_dim)
        self.decoder = build.build_decoder(
                arch = 'ppm_bilinear',
                num_class=19,
                fc_dim=fc_dim,
                use_softmax=False)
    def _init_modules(self):
        resnet_utils.load_pretrained_imagenet_weights(self)

    def forward(self, data):
        pred=self.decoder(self.encoder(data, return_feature_maps=True))
        
        pred = nn.functional.interpolate(
                pred, size=[240,240],
                mode='bilinear', align_corners=False)
        pred = nn.functional.log_softmax(pred, dim=1)
        return pred

def dataloader(bs, gpus):
    roidb, ratio_list, ratio_index = \
            combined_roidb_for_training_semseg('cityscapes_semseg_train')
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


# cfg_file = 'configs/baselines/e2e_segdisp-50_2x.yaml'
cfg_file = 'configs/baselines/e2e_pspnet-101_2x.yaml'
cfg_from_file(cfg_file)
print (cfg.SEM)
print (cfg.DISP)
#cfg_from_list(cfg_file)
#assert_and_infer_cfg()
devices_ids=[0]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(ids) for ids in devices_ids])
# torch.backends.cudnn.benchmark=True
#torch.cuda.set_device(3)
len_gpus = len(devices_ids)
batch_size = 2 * len_gpus
# net = mynn.DataParallel(load_net().to('cuda'), minibatch=True)
# net = mynn.DataParallel(Generalized_SEMSEG().to('cuda'), minibatch=True)
pretrained_model = '/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/output/pspnet50_2gpu_single_scale/Oct20-12-41-16_localhost.localdomain/ckpt/model_17_1486.pth'
checkpoint = torch.load(pretrained_model)
net = Generalized_SEMSEG()
net.load_state_dict(checkpoint['model'])
net = mynn.DataParallel(net.to('cuda'), minibatch=True)
# net = Generalized_SEMSEG()
# net.load_state_dict(checkpoint['model'])
net.train() ##
# optimizer = optim.SGD(net.parameters(), lr=0.000875, momentum=0.9)
# criterion = nn.NLLLoss(ignore_index=255)
dataloader= dataloader(batch_size, len_gpus)
for i, input_data in zip(range(3), dataloader):
    print (input_data.keys())
    # if cfg.DISP.DISP_ON:
    #     input_data['data'] = list(map(lambda x,y: torch.cat((x,y), dim=0),
    #                             input_data['data'], input_data['data_R']))
    #     del input_data['data_R']
    for key in input_data:
        if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
            input_data[key] = list(map(lambda x: Variable(x).to('cuda'), input_data[key]))
    # optimizer.zero_grad()
    loss=net(**input_data)
    print(loss)
    #loss = criterion(pred, label)
    #loss.backward()
    # optimizer.step()
    # print (loss['losses']['loss_semseg'].item())
    # print (loss['losses']['loss_disp'].item())

