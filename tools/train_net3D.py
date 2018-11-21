""" Training Script """
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
from modeling.model_builder_semseg import Generalized_SEMSEG
from modeling.model_builder_3DSD import Generalized_3DSD
from modeling.model_builder_segdisp import Generalized_SEGDISP
from modeling.Loss import Loss
from modeling.pspnet_dispSeg import DispSeg,NetStructure
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch_semseg
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
pynvml.nvmlInit()                                                                                                                                                 
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo= pynvml.nvmlDeviceGetMemoryInfo(handle)



def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--device_ids',
        help='list of gpu device',
        nargs='+', default=[0,1,2,3,4,5,6,7], type=int)

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=100, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_epochs',
        help='Epochs to decay the learning rate on. '
             'Decay happens on the beginning of a epoch. '
             'Epoch is 0-indexed.',
        default=[4, 5], nargs='+', type=int)

    # Epoch
    parser.add_argument(
        '--start_iter',
        help='Starting iteration for first training epoch. 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--start_epoch',
        help='Starting epoch count. Epoch is 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--epochs', dest='num_epochs',
        help='Number of epochs to train',
        default=6, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save in each epoch. '
             'Not include the one at the end of an epoch.',
        default=3, type=int)

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


def main():
    saveNetStructure=False
    
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)


    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        #set gpu device
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ids) for ids in args.device_ids])
        torch.backends.cudnn.benchmark=True
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "cityscapes_semseg_train":
        cfg.TRAIN.DATASETS = ('cityscapes_semseg_train', )
        cfg.MODEL.NUM_CLASSES = 19
    elif args.dataset == "cityscapes":
        cfg.TRAIN.DATASETS = ('cityscape_train_on_val', )
        cfg.MODEL.NUM_CLASSES = 19
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    

    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    print('Batch size change from {} (in config file) to {}'.format(
        original_batch_size, args.batch_size))
    print('NUM_GPUs: %d, TRAIN.IMS_PER_BATCH: %d' % (cfg.NUM_GPUS, cfg.TRAIN.IMS_PER_BATCH))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    ### Adjust learning based on batch size change linearly
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch size change: {} --> {}'.format(
        old_base_lr, cfg.SOLVER.BASE_LR))

    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma

    timers = defaultdict(Timer)
    
    ### Dataset ###
    timers['roidb'].tic()
    if cfg.SEM.SEM_ON or cfg.DISP.DISP_ON:
        roidb, ratio_list, ratio_index = combined_roidb_for_training_semseg(
            cfg.TRAIN.DATASETS)
    else:
        roidb, ratio_list, ratio_index = combined_roidb_for_training(
            cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    train_size = len(roidb)
    logger.info('{:d} roidb entries'.format(train_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    sampler = MinibatchSampler(ratio_list, ratio_index)
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch_semseg if cfg.SEM.SEM_ON or cfg.DISP.DISP_ON else collate_minibatch)

    assert_and_infer_cfg()
    #for args.step, input_data in zip(range(100), dataloader):
    #    data_L = input_data['data']
    #    data_R = input_data['data_R']
    #    label = input_data['disp_label_0']
    #    cv2.imwrite('ims_L.png', data_L[0].numpy()[0].transpose(1,2,0)[:,:,::-1]+cfg.PIXEL_MEANS)
    #    cv2.imwrite('ims_R.png', data_R[0].numpy()[0].transpose(1,2,0)[:,:,::-1]+cfg.PIXEL_MEANS)
    #    cv2.imwrite('label.png', label[0].numpy()[0])
    #    return 
    ### Model ###
    dispSeg=DispSeg()

    if cfg.CUDA:
        dispSeg.to('cuda')

 
    pspnet_bias_params=[]
    pspnet_nonbias_params=[]
    for key,value in dict(dispSeg.pspnet.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                pspnet_bias_params.append(value)
            else:
                pspnet_nonbias_params.append(value)

    pspnet_params=[
        {'params':pspnet_nonbias_params,
         'lr':cfg.SOLVER.BASE_LR,
         'weight_decay':cfg.SOLVER.WEIGHT_DECAY},
        {'params':pspnet_bias_params,
         'lr':cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay':cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0}
    ]


    glassGCN_bias_params=[]
    glassGCN_nonbias_params=[]
    for key,value in dict(dispSeg.glassGCN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                glassGCN_bias_params.append(value)
            else:
                glassGCN_nonbias_params.append(value)

    segdisp3d_params=[
        {'params':glassGCN_nonbias_params,
         'lr':cfg.SOLVER.BASE_LR,
         'weight_decay':cfg.SOLVER.WEIGHT_DECAY},
        {'params':glassGCN_bias_params,
         'lr':cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay':cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0}
    ]


    if cfg.SOLVER.TYPE == "SGD":
        optimizerP = torch.optim.SGD(pspnet_params, momentum=cfg.SOLVER.MOMENTUM)
        optimizerS = torch.optim.SGD(segdisp3d_params, momentum=cfg.SOLVER.MOMENTUM)

    elif cfg.SOLVER.TYPE == "Adam":
        optimizerP = torch.optim.Adam(pspnet_params)
        optimizerS = torch.optim.Adam(segdisp3d_params)

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(pspnet, checkpoint['model'])
        net_utils.load_ckpt(segdisp3d, checkpoint['model'])

        if args.resume:
            assert checkpoint['iters_per_epoch'] == train_size // args.batch_size, \
                "iters_per_epoch should match for resume"
            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            # optimizer.load_state_dict(checkpoint['optimizer'])
            misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
            if checkpoint['step'] == (checkpoint['iters_per_epoch'] - 1):
                # Resume from end of an epoch
                args.start_epoch = checkpoint['epoch'] + 1
                args.start_iter = 0
            else:
                # Resume from the middle of an epoch.
                # NOTE: dataloader is not synced with previous state
                args.start_epoch = checkpoint['epoch']
                args.start_iter = checkpoint['step'] + 1
        del checkpoint
        torch.cuda.empty_cache()



    lr = optimizerP.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    dispSeg = mynn.DataParallel(dispSeg, cpu_keywords=['im_info', 'roidb'], minibatch=True)
    
    ### Training Setups ###
    args.run_name = misc_utils.get_run_name()
    output_dir = misc_utils.get_output_dir(args, args.run_name)
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            #from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    ### Training Loop ###
    dispSeg.train()
    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None)

    iters_per_epoch = int(train_size / args.batch_size)  # drop last
    args.iters_per_epoch = iters_per_epoch
    ckpt_interval_per_epoch = iters_per_epoch // args.ckpt_num_per_epoch
    try:
        logger.info('Training starts !')
        args.step = args.start_iter
        global_step = iters_per_epoch * args.start_epoch + args.step
        for args.epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
            # ---- Start of epoch ----

            # adjust learning rate
            if args.lr_decay_epochs and args.epoch == args.lr_decay_epochs[0] and args.start_iter == 0:
                args.lr_decay_epochs.pop(0)
                net_utils.decay_learning_rate(optimizerP, lr, cfg.SOLVER.GAMMA)
                net_utils.decay_learning_rate(optimizerS, lr, cfg.SOLVER.GAMMA)
                lr *= cfg.SOLVER.GAMMA

            for args.step, input_data in zip(range(args.start_iter, iters_per_epoch), dataloader):
                
                if cfg.DISP.DISP_ON:
                    input_data['data'] = list(map(lambda x,y: torch.cat((x,y), dim=0), 
                                input_data['data'], input_data['data_R']))
                    if cfg.SEM.DECODER_TYPE.endswith('3Ddeepsup'):
                        input_data['disp_scans'] = torch.arange(0,
                                cfg.DISP.MAX_DISPLACEMENT).float().view(1,cfg.DISP.MAX_DISPLACEMENT,1,1).repeat(args.batch_size,1,1,1)
                    del input_data['data_R']

                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(lambda x: Variable(x, requires_grad=False).to('cuda'), input_data[key]))
                training_stats.IterTic()
                net_outputs=dispSeg(**input_data)
                training_stats.UpdateIterStats(net_outputs)
                #loss = net_outputs['losses']['loss_semseg']
                #acc  = net_outputs['metrics']['accuracy_pixel']
                #print (loss.item(), acc)
                #for key in net_outputs.keys():
                #    print(key)
                loss = net_outputs['total_loss']
                
                #print("loss.shape:",loss)
                optimizerP.zero_grad()
                optimizerS.zero_grad()
                loss.backward()
                optimizerP.step()
                optimizerS.step()
                training_stats.IterToc()

                if args.step % args.disp_interval == 0:
                    #disp_image=net_outputs['disp_image']
                    #semseg_image=net_outputs['semseg_image']
                    #tblogger.add_image('disp_image',disp_image,global_step)
                    #tblogger.add_image('semseg_image',semseg_image,global_step)
                    log_training_stats(training_stats, global_step, lr)
                global_step += 1
            # ---- End of epoch ----
            # save checkpoint
            
            net_utils.save_ckpt(output_dir, args, dispSeg, optimizerS)
            # reset starting iter number after first epoch
            args.start_iter = 0

        # ---- Training ends ----
        #if iters_per_epoch % args.disp_interval != 0:
            # log last stats at the end
        #    log_training_stats(training_stats, global_step, lr)

    except (RuntimeError, KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        net_utils.save_ckpt(output_dir, args, dispSeg, optimizerS)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()


def log_training_stats(training_stats, global_step, lr):
    stats = training_stats.GetStats(global_step, lr)
    log_stats(stats, training_stats.misc_args)
    if training_stats.tblogger:
        training_stats.tb_log_stats(stats, global_step)


if __name__ == '__main__':
    main()
