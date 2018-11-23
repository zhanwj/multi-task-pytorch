#/scp -P 1234 -r ./lib/modeling/model_builder_3DSD.py zhanwj@139.219.68.155:/mnt/data/zhanwj/Detectron.pytorch/lib/modeling
scp -r ./lib/modeling/ longchen@222.200.180.178:/home/longchen/Detectron.pyTorch/lib/
#scp-rr ./lib/roi_data zhanwj@139.219.68.155:/mnt/data/zhanwj/Detectron.pytorch/lib
scp -r ./lib/core longchen@222.200.180.178:/home/longchen/Detectron.pyTorch/lib
scp -r ./configs/baselines longchen@222.200.180.178:/home/longchen/Detectron.pyTorch/configs
scp -r ./tools longchen@222.200.180.178:/home/longchen/Detectron.pyTorch
#scp -P 1234 -r ./lib/utils zhanwj@139.219.68.155:/mnt/data/zhanwj/Detectron.pytorch/lib
#scp -P 1234 -r ./conv3dTest.py zhanwj@139.219.68.155:/mnt/data/zhanwj/Detectron.pytorch/

