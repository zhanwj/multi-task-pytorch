LOG=psp_deepsup_step_42epoch_2gpus_device03.log
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
CUDA_VISIBLE_DEVICES=7 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python  -u tools/train_net_Nov_13.py --dataset cityscapes \
    --cfg configs/baselines/e2e_segdisp-50_2x.yaml \
    --bs 1 --nw 1  --device_ids 7 \
    --epochs 480 --ckpt_num_per_epoch 9 \
    --lr_decay_epochs 24 30 \
    --disp_interval 5  --no_save --use_tfboard
