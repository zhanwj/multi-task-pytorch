LOG=psp_deepsup_step_42epoch_2gpus_device03.log
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
CUDA_VISIBLE_DEVICES=4 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python  -u tools/train_net_cspn.py --dataset cityscapes_semseg_train \
    --cfg configs/baselines/e2e_segdisp-R-50_cspn_1x.yaml \
    --bs 1 --nw 4  --device_ids 1 2 \
    --epochs 480 --ckpt_num_per_epoch 9 \
    --lr_decay_epochs 24 30 \
    --disp_interval 5 --no_save  --use_tfboard
