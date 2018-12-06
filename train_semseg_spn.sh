LOG=psp_deepsup_step_42epoch_2gpus_device03.log
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
CUDA_VISIBLE_DEVICES=5 /home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python  -u tools/train_net.py --dataset cityscapes \
    --cfg configs/baselines/e2e_segsem-fcn8s_spn.yaml \
    --bs 1 --nw 1  --device_ids 3 4 \
    --epochs 400 --ckpt_num_per_epoch 20 \
    --lr_decay_epochs 20 30 40 50 \
    --disp_interval 20 --no_save  --use_tfboard
