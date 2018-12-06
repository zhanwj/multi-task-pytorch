LOG=psp_deepsup_step_42epoch_2gpus_device03.log
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
CUDA_VISIBLE_DEVICES=6 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python  -u tools/train_net_psp_pretrained_test.py --dataset cityscapes \
    --cfg configs/baselines/e2e_segcspn-50_2x.yaml \
    --bs 2 --nw 1  --device_ids 3 4 \
    --epochs 40 --ckpt_num_per_epoch 10 \
    --lr_decay_epochs 20 30 40 50 \
    --disp_interval 5 --no_save  --use_tfboard
