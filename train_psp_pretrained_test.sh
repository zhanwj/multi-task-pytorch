LOG=psp_deepsup_step_42epoch_2gpus_device03.log
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
#--cfg configs/baselines/e2e_psp_pretrained_test.yaml  \
CUDA_VISIBLE_DEVICES=5,6 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python  -u tools/train_net_psp_pretrained_test.py --dataset cityscapes \
    --cfg configs/baselines/e2e_psp_pretrained_test.yaml \
    --bs 4 --nw 2  --device_ids 3 4 \
    --epochs 40   --ckpt_num_per_epoch 9 \
    --lr_decay_epochs 24 30 \
    --disp_interval 20  --use_tfboard
