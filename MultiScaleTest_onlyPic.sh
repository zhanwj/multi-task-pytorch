val_pic_path="seg_pred_pic/pred_sem_val_500_pspnet50_plateau_ms"
val_pic_labelId="_labelId"
export export PYTHONPATH=/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin:$PYTHONPATH
/home/zhanwj/Desktop/pyTorch/anaconda3/envs/pytorch/bin/python -u test_multiscale.py --dataset cityscapes \
    --config configs/baselines/e2e_pspnet-101_2x.yaml \
    --gpu 6 \
    --save_file ${val_pic_path} \
    --network Generalized_SEMSEG \
    --pretrained_model output/pspnet50_multiscale_ReduceLROnPlateau_40epochs_720/e2e_pspnet-101_2x/Dec07-11-03-03_localhost.localdomain/ckpt//model_58_1486.pth\
    --index_start 0 --index_end 500