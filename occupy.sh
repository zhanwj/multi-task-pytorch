LOG=psp_deepsup_step_42epoch_2gpus_device03.log
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
CUDA_VISIBLE_DEVICES=3 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python   tools/occupy.py --dataset cityscapes \
    --cfg configs/baselines/e2e_segdisp-50_2x.yaml\
    --bs 2 --nw 4  --device_ids 1 \
    --epochs 3600 --ckpt_num_per_epoch 9 \
    --lr_decay_epochs 24 30 \
    --disp_interval 20  --no_save  --use_tfboard
