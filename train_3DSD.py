LOG=psp_deepsup_step_42epoch_2gpus_device03.log
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
CUDA_VISIBLE_DEVICES=5 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python  -u tools/train_net3DSD.py --dataset cityscapes \
    --cfg configs/baselines/e2e_segdisp-R-50_3Dpool_1x_SD.yaml \
    --bs 2 --nw 2  --device_ids 5 \
    --epochs 480 --ckpt_num_per_epoch 9 \
    --lr_decay_epochs 24 30 \
    --disp_interval 5 --no_save  --use_tfboard
