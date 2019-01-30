""" Training Script """
from tensorboardX import SummaryWriter
import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict
import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
#cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training_semseg
from modeling.model_builder_semseg_bat import Generalized_SEMSEG
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch_semseg
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats
import lib.utils.segm_data as torchdata


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
#define start for poly
poly_iter=0

def group_weight(module):
    group_decay = []
    group_no_decay = []
    keep_bn = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, mynn.AffineChannel2d):
            if m.weight is not None:
                if not cfg.SEM.FREEZE_BN:
                    group_no_decay.append(m.weight)
                else:
                    keep_bn += 1
            if m.bias is not None:
                if not cfg.SEM.FREEZE_BN:
                    group_no_decay.append(m.bias)
                else:
                    keep_bn += 1
		
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                if not cfg.SEM.FREEZE_BN:
                    keep_bn+=1
                else:
                    group_no_decay.append(m.weight)
            if m.bias is not None:
                if not cfg.SEM.FREEZE_BN:
                    keep_bn+=1
                else:
                    group_no_decay.append(m.bias)

    #assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)+keep_bn
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(nets, args):
    net_decoder = nets
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer_decoder


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** cfg.SOLVER.POLY_POWER)
    args.running_lr_decoder = cfg.SOLVER.BASE_LR * scale_running_lr
    optimizer_decoder = optimizers
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = args.running_lr_decoder
    return args.running_lr_decoder

def step_adjust_learning_rate(optimizers, lr, cur_iter, args):
    if args.lr_decay_epochs:
        if args.epoch == args.lr_decay_epochs[0] and args.start_iter == 0:
            args.lr_decay_epochs.pop(0)
            lr *= cfg.SOLVER.GAMMA
            for param_group in optimizer_decoder.param_groups:
                param_group['lr'] = lr
        global poly_iter
        poly_iter = cur_iter
        return lr
    else:
        scale_running_lr = ((1. - float(cur_iter-poly_iter) / (args.max_iters-poly_iter)) ** cfg.SOLVER.POLY_POWER)
        args.running_lr_decoder = cfg.SOLVER.BASE_LR * scale_running_lr
        optimizer_decoder = optimizers
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = args.running_lr_decoder
        return args.running_lr_decoder


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
    parser.add_argument(
        '--epoch_iters', default=5000, type=int,
        help='iterations of each epoch (irrelevant to batch size)')

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
    elif args.dataset == "cityscapes":
        cfg.TRAIN.DATASETS = ('cityscapes_semseg_train', )
        cfg.MODEL.NUM_CLASSES = 19
    elif args.dataset == "cityscape_train_on_val":
        cfg.TRAIN.DATASETS = ('cityscape_train_on_val', )
        cfg.MODEL.NUM_CLASSES = 19
    elif args.dataset == "cityscapes_coarse":
        cfg.TRAIN.DATASETS = ('cityscapes_coarse', )
        cfg.MODEL.NUM_CLASSES = 19
    elif args.dataset == "cityscapes_trainval":
        cfg.TRAIN.DATASETS = ('cityscapes_trainval', )
        cfg.MODEL.NUM_CLASSES = 19
    elif args.dataset == "cityscapes_all":
        cfg.TRAIN.DATASETS = ('cityscapes_all', )
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

    #sampler = MinibatchSampler(ratio_list, ratio_index)
    sampler = None
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    #dataloader = torch.utils.data.DataLoader(
    #    dataset,
    #    batch_size=args.batch_size,
    #    sampler=sampler,
    #    num_workers=cfg.DATA_LOADER.NUM_THREADS,
    #    collate_fn=collate_minibatch_semseg if cfg.SEM.SEM_ON or cfg.DISP.DISP_ON else collate_minibatch,
    #    drop_last=False,
    #    shuffle=True) # when load image will be shuffle in each epoch
    ## Dataset and Loader
    #dataset_train = TrainDataset(
    #    args.list_train, args, batch_per_gpu=args.batch_size_per_gpu)
    #args.epoch_iters=dataset_train.num_sample//(args.num_gpus*args.batch_size_per_gpu)
    dataloader = torchdata.DataLoader(
        dataset,
        batch_size=args.batch_size,  # we have modified data_parallel
        collate_fn=collate_minibatch_semseg,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        drop_last=True,
        shuffle=True,
        maxsize=cfg.TRAIN.IMS_PER_BATCH*2)

    assert_and_infer_cfg()
    #for data in dataloader:
    #    image = data['data'][0][0].numpy()
    #    print (image.shape)
    #    image=image.transpose(1,2,0)+cfg.PIXEL_MEANS
    #    cv2.imwrite('image.png', image[:,:,::-1])
    #    cv2.imwrite('label.png',10*data['semseg_label_0'][0][0].numpy())
    #    return
    
    maskRCNN = eval(cfg.MODEL.TYPE)()
    if len(cfg.SEM.PSPNET_PRETRAINED_WEIGHTS)>1:
        print("loading pspnet weights")
        state_dict={}
        pretrained=torch.load(cfg.SEM.PSPNET_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
        pretrained = pretrained['model']
        if cfg.SEM.SPN_ON:
            maskRCNN.pspnet.load_state_dict(pretrained,strict=True)
        #elif 'deform_conv' in cfg.MODEL.CONV_BODY or 'deeplab' in cfg.SEM.DECODER_TYPE:
        elif 'deeplab' in cfg.SEM.DECODER_TYPE and 'uber' in cfg.SEM.PSPNET_PRETRAINED_WEIGHTS:
            encoder=dict()
            for k, v in pretrained.items():
                if 'decoder' in k:
                    continue
                encoder[k.replace('encoder.','')] = v
            maskRCNN.encoder.load_state_dict(encoder,strict=False)
        else:
            maskRCNN.load_state_dict(pretrained,strict=False)
        print("weights load success")

    if cfg.SEM.SPN_ON:
        maskRCNN.pspnet.eval()
        for p in maskRCNN.pspnet.parameters():
            p.requires_grad = False

    if cfg.CUDA:
        maskRCNN.to('cuda')
    #print(maskRCNN)
    ### Optimizer ###
    bias_params = []
    nonbias_params = []
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                nonbias_params.append(value)
    params = [
        {'params': nonbias_params,
         'lr': cfg.SOLVER.BASE_LR,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0}
    ]


    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
        print("Using STEP as Lr reduce policy!")
    if cfg.SOLVER.TYPE == 'SGD' and cfg.SOLVER.LR_POLICY == 'ReduceLROnPlateau':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10)
        print("Using ReduceLROnPlateau as Lr reduce policy!")
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)
    elif "poly" in cfg.SOLVER.TYPE:
        optimizer = create_optimizers(maskRCNN,args)
        print("Using Poly as Lr reduce policy!")

    args.max_iters = (int(train_size / args.batch_size)) * args.num_epochs
    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])
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

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)
    
    if cfg.SOLVER.TYPE=='step_poly':
        lr  = cfg.SOLVER.BASE_LR / (cfg.SOLVER.GAMMA**len(args.lr_decay_epochs))
    else:
        lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.


    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True)
	
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
    maskRCNN.train()

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
            if args.lr_decay_epochs and args.epoch == args.lr_decay_epochs[0] and args.start_iter == 0 and cfg.SOLVER.LR_POLICY=='steps_with_decay' :
                args.lr_decay_epochs.pop(0)
                net_utils.decay_learning_rate(optimizer, lr, cfg.SOLVER.GAMMA)
                lr *= cfg.SOLVER.GAMMA

            

            for args.step, input_data in zip(range(args.start_iter, iters_per_epoch), dataloader):
                
                if cfg.DISP.DISP_ON:
                    input_data['data'] = list(map(lambda x,y: torch.cat((x,y), dim=0), 
                                input_data['data'], input_data['data_R']))
                    if cfg.SEM.DECODER_TYPE.endswith('3D'):
                        input_data['disp_scans'] = torch.arange(1,
                                cfg.DISP.MAX_DISPLACEMENT+1).float().view(1,cfg.DISP.MAX_DISPLACEMENT).repeat(args.batch_size,1)
                    del input_data['data_R']

                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(lambda x: Variable(x, requires_grad=False).to('cuda'), input_data[key]))
                training_stats.IterTic()
                net_outputs = maskRCNN(**input_data)
                training_stats.UpdateIterStats(net_outputs)
                #loss = net_outputs['losses']['loss_semseg']
                #acc  = net_outputs['metrics']['accuracy_pixel']
                #print (loss.item(), acc)
                #for key in net_outputs.keys():
                #    print(key)
                loss = net_outputs['total_loss']
                
                #print("loss.shape:",loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cfg.SOLVER.TYPE=='poly':
                    lr = adjust_learning_rate(optimizer, global_step, args)
                
                if cfg.SOLVER.TYPE=='step_poly':
                    lr = step_adjust_learning_rate(optimizer, lr, global_step, args)
                
                training_stats.IterToc()

                if args.step % args.disp_interval == 0:
                    disp_image=''
                    semseg_image=''
                    #tblogger.add_image('disp_image',disp_image,global_step)
                    #tblogger.add_image('semseg_image',semseg_image,global_step)
                    log_training_stats(training_stats, global_step, lr)
                global_step += 1
            # ---- End of epoch ----
            # save checkpoint
            if cfg.SOLVER.TYPE == 'SGD' and cfg.SOLVER.LR_POLICY == 'ReduceLROnPlateau':
                    lr_scheduler.step(loss)
                    lr = optimizer.param_groups[0]['lr']
            if (args.epoch+1) % args.ckpt_num_per_epoch ==0:
                net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)
            # reset starting iter number after first epoch
            args.start_iter = 0

        # ---- Training ends ----
        #if iters_per_epoch % args.disp_interval != 0:
            # log last stats at the end
        #    log_training_stats(training_stats, global_step, lr)
        # save final model
        if (args.epoch+1) % args.ckpt_num_per_epoch:
            net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)
    except (RuntimeError, KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)
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
    
