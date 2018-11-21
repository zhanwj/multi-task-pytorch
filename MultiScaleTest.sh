gpu_list={0, 1, 2}
for gpu in gpu_list:
nohup python test_multiscale.py cityscapes pred_semseg \
    --cfg filename
    --gpu $gpu
    --index_start 0 --index_end 500
    --input_size 720 720
    --aug_scales 1440 &
