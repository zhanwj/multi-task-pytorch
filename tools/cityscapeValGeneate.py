from scipy.misc import imread
import random
import os
import copy
random.seed(1)
listfile='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/annotations/val.txt'
datafile='/home/zhanwj/Desktop/pyTorch/Detectron.pytorch/lib/datasets/data/cityscapes/'
trainlist=[]
vallist=[]
used=[]
trainCount=[0]*19
valCount=[0]*19
countlist=[0 for i in range(19)]
#print(len(countlist))
def loadImage(image):
    img=imread(image)
    img=img.reshape(1,img.size)
    sortList=sorted(list(set(img[0].tolist())))
    return sortList

#trainid=[[0 for col in range(19)] for row in range(200)]
#trainid=[0]*19
indexList=[line.rstrip('\n') for line in open(listfile)]
indexListOri=indexList
random.shuffle(indexList)
def toTrainedIdList(index,trainid):
    #temp=[]
    _,_,disp,_=index.split()
    img=loadImage(os.path.join(datafile,disp))
    for i in range(len(img)):
        if img[i]<200:
            trainid[img[i]].insert(0,index)

def toList(index):
    temp=[[0 for col in range(19)] for row in range(200)]
    for i in range(len(index)):
        toTrainedIdList(index[i],temp)
    for i in range(19):
        while temp[i][len(temp[i])-1]==0:
            _= temp[i].pop()
    return temp

trainid=toList(indexList)
trainidOriginal=copy.deepcopy(trainid)
#for i in range(19):
   # print(len(trainid[i]))


#for i in range(len(indexList)):
#    toTrainedIdList(indexList[i])
#for i in range(19):
#    while trainid[i][len(trainid[i])-1]==0:
#        _= trainid[i].pop()
#    print(len(trainid[i]))
def updateList():
    for i in range(19):
        countlist[i]=len(trainid[i])
    return countlist

countlist=updateList()
#for i in range(19):
   # print(countlist[i])
def deleteTrainid(name):
    for i in range(19):
        if name in trainid[i]:
            trainid[i].remove(name)



ids=countlist.index(min(countlist))
for i in range(19):
    countlist=updateList()
    while ids in used:
        countlist[ids]=255
        ids=countlist.index(min(countlist))
        #used.append(ids)
        #used[i]=ids
    used.append(ids)
    num=len(trainid[ids])
    trainNum=0.7*num
    valNum=num-trainNum
    random.shuffle(trainid[ids])
    for j in range(len(trainid[ids])):
        name=trainid[ids].pop()
        if j<=trainNum:
            trainlist.append(name)
        else:
            vallist.append(name)
        deleteTrainid(name)
   # print(used)
print("trainNum:",len(trainlist))
print("valNum:",len(vallist))

trainIdlist=toList(trainlist)
valIdlist=toList(vallist)

f=open('Cityscape_disp_SegFlow_train.txt','w+')                                                                                                        
g=open('Cityscape_disp_SegFlow_val.txt','w+')
h=open('Cityscape_disp_SegFlow_train_val_count.txt','w+')

for i in range(19):
    so=str(len(trainidOriginal[i]))
    st=str(len(trainIdlist[i]))
    sv=str(len(valIdlist[i]))
    s="trainID:"+str(i)+" total:"+so+" train:"+st+" val:"+sv+'\n'
    h.write(s)
for i in range(len(trainlist)):
    f.write(trainlist[i]+'\n')
for i in range(len(vallist)):
    g.write(vallist[i]+'\n')

h.close()
f.close()
g.close()











