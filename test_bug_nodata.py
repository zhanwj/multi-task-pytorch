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
    inputs = {}
    inputs['data'] = Variable(torch.randn(2*bs*gpus, 3, 240, 240)).to('cuda')
    inputs['semseg_label_0'] = Variable(torch.LongTensor(
        np.random.randint(0, 19, (bs*gpus, 240, 240), dtype=np.long))).to('cuda')
    inputs['disp_label_0'] = Variable(torch.rand(bs*gpus,  240, 240)).to('cuda')
    return inputs

cfg_file = 'configs/baselines/e2e_segdisp-50_2x.yaml'
cfg_from_file(cfg_file)
print (cfg.SEM)
print (cfg.DISP)
#cfg_from_list(cfg_file)
#assert_and_infer_cfg()
devices_ids=[7]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(ids) for ids in devices_ids])
torch.backends.cudnn.benchmark=True
#torch.cuda.set_device(3)
len_gpus = len(devices_ids)
batch_size = 2 * len_gpus
#net = mynn.DataParallel(load_net().to('cuda'), minibatch=True)
net = mynn.DataParallel(Generalized_SEGDISP().to('cuda'), minibatch=True)
optimizer = optim.SGD(net.parameters(), lr=0.000875, momentum=0.9)
criterion = nn.NLLLoss(ignore_index=255)
#dataloader= dataloader(batch_size, len_gpus)
for i in range(10):
#for i, inputs in zip(range(1000), dataloader):
    inputs = dataloader(batch_size, len_gpus)
    for key in inputs:
        print("before")
        print(key,inputs[key].shape)
        
        inputs[key] = torch.chunk(inputs[key], chunks=len_gpus, dim=0)
        print("after:")
        print(key,inputs[key].shape)
    optimizer.zero_grad()
    loss=net(**inputs)
    #loss = criterion(pred, label)
    #loss.backward()
    optimizer.step()
    print (loss['losses']['loss_semseg'].item())
    print (loss['losses']['loss_disp'].item())

