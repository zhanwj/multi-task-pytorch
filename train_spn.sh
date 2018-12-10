#LOG=psp_deepsup_step_42epoch_2gpus_device03.log
#export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
#--cfg configs/baselines/e2e_pspnet-SE_R50_2x.yaml \
CUDA_VISIBLE_DEVICES=7 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python  -u tools/train_net_psp_pretrained_test.py --dataset cityscapes \
    --cfg configs/baselines/e2e_spn-101_2x.yaml \
    --bs 1 --nw 2  --device_ids 3 4 \
    --epochs 60 --ckpt_num_per_epoch 9 \
    --lr_decay_epochs 24 30 \
    --disp_interval 20  --no_save --use_tfboard
