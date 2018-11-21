import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch_semseg
from datasets.roidb import combined_roidb_for_training, combined_roidb_for_training_semseg
import os
import numpy as np
import nn as mynn
import cv2
from modeling.model_builder_segdisp import Generalized_SEGDISP
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from tensorboardX import SummaryWriter
import torchvision
from lib.modeling import CRL
net = CRL.DispResNet()
left=torch.Tensor(1,3,640,640)
right=torch.Tensor(1,3,640,640)
disp=torch.Tensor(1,1,640,640)
inputs=[]
inputs.append(left)
inputs.append(right)
inputs.append(disp)
writer=SummaryWriter(log_dir='./network_structure')
#os.environ["CUDA_VISIBLE_DEVICES"] = 7
with writer:
	writer.add_graph(net,inputs)