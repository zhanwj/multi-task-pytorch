import os
import random
list_txtpath='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/label_info_coarse/train.txt'
save_path='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/annotations/'
coarse_train='coarse_train.txt'
data_path='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/'
fine_train ="train.txt" 
coarse_fine_mixed="coarse_fine_mixed"
coarse_lines=[]
coarse_fine_mixed_list=[]

def readlines(filename):
    temp=[]
    with open(filename,'r') as f:
        for lines in f.readlines():
            temp.append(lines)
    return temp

def writelines(filename,list_to_wirte):
    with open(filename,'w') as f:
        for i in range(len(list_to_wirte)):
            f.write(list_to_wirte[i])

def check_file(file_path):
    coarse=readlines(file_path)
    fail=0
    for i in range(len(coarse)):
        left,right,sem,_=coarse[i].split()
        if not os.path.isfile(os.path.join(data_path,left)):
            print("file does not exist:{}".format(left))
            fail=fail+1
        if not os.path.isfile(os.path.join(data_path,sem)):
            print("file does not exist:{}".format(sem))
            fail=fail+1
        print("Number of samples had been processed:{}".format(i))
        print("File not exist:",fail)

with open(list_txtpath,'r') as f:
    for lines in f.readlines():
        line=lines.split()
        left=line[0][75:]
        right='right'
        sem=line[1][75:]
        disparity='disparity'
        coarse_line=left+' '+right+' '+sem+' '+disparity+'\n'
        coarse_lines.append(coarse_line)
print("Number of samples had been processed:{}".format(len(coarse_lines)))
f2=open(os.path.join(save_path,coarse_train),'w')
f2.writelines(coarse_lines)
f2.close()

coarse_train_list=readlines(os.path.join(save_path,coarse_train))
fine_train_list=readlines(os.path.join(save_path,fine_train))
for i in range(4):
    coarse_train_list.extend(fine_train_list)
print("Total Number of samples:",len(coarse_train_list))
random.shuffle(coarse_train_list)

writelines(os.path.join(save_path,coarse_fine_mixed),coarse_train_list)

check_file(os.path.join(save_path,coarse_train))
