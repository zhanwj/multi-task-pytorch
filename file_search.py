local_list_path=''
remote_list_path=''
local_save_path='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/annotations'
remote_save_path='/data4/chenlong2/zhanwj/zhanwj/Detectron.pytorch/datasets/cityscapes/dataSet/cityscapes/annotations'
local_save_name='label_test.txt'
remote_save_name="completed_label.txt"
local_rootdir="/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes"
local_label_name="/gtCoarse/train_extra"
local_left_name="leftImg8bit_trainvaltest/leftImg8bit"
remote_label_path="/data4/chenlong2/zhanwj/zhanwj/Detectron.pytorch/datasets/cityscapes/dataSet/cityscapes/gtCoarse/train_extra"

left_image="/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
gt_fine="/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/gtFine"
import os
def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) 
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files


            



label_list=list_all_files(gt_fine)
prefix_num=len(local_rootdir)
for i in range(len(label_list)):
    label_list[i]=label_list[i][prefix_num:]
print("contain {} files".format(len(label_list)))

#with open(os.path.join(remote_save_path,remote_save_name),'w') as fp:
#    for item in label_list:
#        fp.write(item+'\n')
#