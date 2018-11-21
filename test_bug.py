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
        #self.conv = nn.Conv2d(3, 1, 1)
        #build = snet.ModelBuilder()
        #encoder = builder.build_encoder(arch='', fc_dim=2048)
        #self._init_modules()
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
    return torch.randn(bs*gpus, 3, 720, 720), \
            torch.LongTensor(np.random.randint(0, 19, (bs*gpus, 90, 90), dtype=np.long))

cfg_file = 'configs/baselines/e2e_pspnet-50_2x.yaml'
cfg_from_file(cfg_file)
#cfg_from_list(cfg_file)
assert_and_infer_cfg()


devices_ids=[1, 2]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(ids) for ids in devices_ids])
torch.backends.cudnn.benchmark=True
#torch.cuda.set_device(3)
len_gpus = len(devices_ids)
batch_size = 2 * len_gpus
#net = mynn.DataParallel(load_net().to('cuda'), minibatch=True)
net = mynn.DataParallel(Generalized_SEGDISP().to('cuda'), minibatch=True)
optimizer = optim.SGD(net.parameters(), lr=0.000875, momentum=0.9)
criterion = nn.NLLLoss(ignore_index=255)
dataloader= dataloader(batch_size, len_gpus)
#for i in range(10):
for i, inputs in zip(range(1000), dataloader):
    #data, label= dataloader(batch_size, len_gpus)
    #data = Variable(data).to('cuda')
    #data  = torch.chunk(data, chunks=len_gpus, dim=0)
    #label = Variable(label).to('cuda')
    #assert torch.all(data >= 0) and torch.all(data < 19), 'label is error'
    for key in inputs:
        inputs[key] = list(map(lambda x:Variable(x).to('cuda'), inputs[key]))
    data, label = inputs['data'], inputs['semseg_label_0']
    #cv2.imwrite('ims.png', data[0].cpu().numpy()[0].transpose(1,2,0)[:,:,::-1])
    label = torch.cat(label, 0)
    #cv2.imwrite('label.png', label.cpu().numpy()[0])
    if len_gpus == 1:
        data = data[0]
    optimizer.zero_grad()
    pred=net(data)
    loss = criterion(pred, label)
    #loss.backward()
    optimizer.step()
    print (loss.item())

