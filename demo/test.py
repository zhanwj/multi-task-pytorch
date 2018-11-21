#encoding=utf-8
#图片处理
import sys
sys.path.append('../python')
sys.path.append('code')
#import optIO
import caffe
#print caffe.__file__
import numpy as np
import time
import os
import scipy.io as scio
import loadpath as lp
import scipy.misc as smi
import cv2
from caffe.io import Transformer
import logging as logger
from labels import labels
import pickle
import time
from PIL import Image
#logger.basicConfig(format='Process:',level=logger.DEBUG)

def write(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
def load(filename):
    with open(filename, 'rb') as fp:
        data=pickle.load(fp)
    return data

def argument():
    import argparse
    parser = argparse.ArgumentParser(description='Process prediction.')
    parser.add_argument('dataset', help='give a dataset name', type=str)
    parser.add_argument('save_file',   help='which file to restore', type=str)
    parser.add_argument('--gpu', dest='gpu', help='give a gpu to train network', default=0, type=int)
    parser.add_argument('--input_size', dest='input_size', help='input size of network', nargs='+', default=[400, 800], type=int)
    #parser.add_argument('--aug_scale', dest='aug_scale', help='scale image of network', default=[0.55,0.6,0.65,0.7], type=list)
    parser.add_argument('--aug_scale', dest='aug_scale', help='scale image of network', nargs='+', default=[0.55, 0.65, 0.7 ,0.75], type=int)
    parser.add_argument('--network', dest='network', help='network name', default='deeplab101', type=str)
    parser.add_argument('--topK', dest='topK', help='test topK image', default=10, type=int)
    args = parser.parse_args()
    return args


class TestNet(object):
    def __init__(self, args):
        self._cur = 0
        if args.dataset == 'cityscapes':
            self.input_size = args.input_size
            self.aug_scale = args.aug_scale
            self.label_root = os.path.join(os.getcwd(),'citycapes/label_info/val_label_citycapes.txt')
            #self.label_root = os.path.join(os.getcwd(),'citycapes/label_info/onlytrain_label_citycapes_right.txt')
            self.num_class = 19
            self.image_shape=[1024, 2048]
            if args.network == 'deeplab101':
                self.deploy_file=lp.deploy_deeplab101
                self.model_weight=lp.deeplab101_final_model
            else:
                self.deploy_file=lp.deploy_deeplab50
                self.model_weight=lp.deeplab50_final_model
        self.load_listname(args)
        transformer = Transformer({'data':(1, 3, self.input_size[0], self.input_size[1])})
        transformer.set_transpose('data',(2,0,1)) ##
        transformer.set_mean('data', np.array([123.68, 116.779, 103.939]))
        transformer.set_raw_scale('data',255)
        transformer.set_channel_swap('data',(2,1,0)) ##
        self.transformer = transformer

        #transformer label
        self.transLabel = {label.trainId : label.id for label in labels} ##
        self.transColor = {label.trainId : label.color for label in labels}

    def load_listname(self, args):
        indexlist = [line.split()[0] for line in open(self.label_root,'r').read().splitlines()]
        self.indexlist = indexlist[: args.topK]

    def load_net(self, args):
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
        self.net=caffe.Net(self.deploy_file, self.model_weight, caffe.TEST)

    def load_image(self, args):
        imgname = self.indexlist[self._cur].split()[0]
        self._cur += 1
        image = Image.open(imgname)
        return image, imgname

    def transfer_img(self, image, imgname, scale, args): #将图片切分成多块
        #image = cv2.imread(imgname)
        #image = cv2.resize(image, None,None,fx=scale,fy=scale, interpolation=cv2.INTER_NEAREST)
        resize_h = int(np.round(scale*self.image_shape[0]))
        resize_w = int(np.round(scale*self.image_shape[1]))
        image = np.array(image.resize((resize_w, resize_h)))
        im_shape = image.shape[:2] #720 1440
        step_h = int(np.ceil(1.0*im_shape[0] / args.input_size[0]))
        step_w = int(np.ceil(1.0*im_shape[1] / args.input_size[1]))
        one_list = []
        tmp_name = os.path.join(args.save_file,imgname.split('/')[-1])
        for ih in range(step_h):
            for iw in range(step_w):
                inputs = np.zeros(args.input_size+[3])
                max_h = min(im_shape[0], (ih+1)*args.input_size[0])
                max_w = min(im_shape[1], (iw+1)*args.input_size[1])
                in_h= max_h - ih*args.input_size[0]
                in_w= max_w - iw*args.input_size[1]
                inputs[0: in_h, 0: in_w] = image[ih*args.input_size[0]: max_h,
                        iw*args.input_size[1]: max_w]
                #cv2.imwrite(tmp_name.replace('.png','%d_%d.png' %(ih, iw)), inputs)
                inputs = self.transformer.preprocess('data', inputs)
                one_list.append(inputs)

        #cv2.imwrite(tmp_name+'_.png', pred_prob[:1024, :2048, :])
        return one_list, imgname.split('/')[-1], [step_h, step_w]

    def save_pred(self, pred_list, image_name, scale_info, index, args): #拼起来
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        step_h, step_w = index
        scale, scale_i = scale_info
        for ih in range(step_h):
            for iw in range(step_w):
                if iw == 0:
                    pred_w = pred_list[ih*step_w]
                else:
                    pred_w = np.concatenate((pred_w, pred_list[ih*step_w+iw]), axis=2)
            if ih == 0:
                pred_prob = pred_w
            else:
                pred_prob = np.concatenate((pred_prob, pred_w), axis=1)
        max_h = int(np.round(self.image_shape[0] * scale))
        max_w = int(np.round(self.image_shape[1] * scale))
        pred_prob = pred_prob[:, 0: max_h, 0: max_w] ## c,h,w
        write( tmp_name+'_'+str(scale_i)+'_prob.pkl', pred_prob)
        #self.transfer_label(image_name, np.argmax(pred_prob, axis=0))

    def save_multi_results(self, image_name, args): #多尺寸
        num_scale = len(self.aug_scale)
        pred_prob = np.zeros([self.num_class]+self.image_shape)
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        for i, scale in enumerate(self.aug_scale):
            scale_pred = load(tmp_name+'_'+str(i)+'_prob.pkl')
            rscale=1.0 / scale
            for ic in range(self.num_class): ##
                pred_prob[ic] += cv2.resize(scale_pred[ic], tuple(self.image_shape[::-1]), interpolation=cv2.INTER_LINEAR)
            #os.remove(tmp_name+'_'+str(scale)+'.txt')
        pred_prob = np.argmax(pred_prob, axis=0)
        self.transfer_label(image_name, pred_prob)

    def transfer_label(self, image_name, pred_prob): #图片可视化： 灰度图-》彩色图
        tmp_name = os.path.join(args.save_file,image_name.replace('.png',''))
        assert np.all(pred_prob >=0) and np.all(pred_prob<19), 'error in prediction'
        colors = np.zeros(list(pred_prob.shape)+ [3], dtype='uint8') # 19 1024 2048 3
        for ic in range(pred_prob.shape[0]):
            for ir in range(pred_prob.shape[1]):
                colors[ic, ir, :] = self.transColor[ pred_prob[ic, ir] ]
        Idsave = Image.fromarray(colors.astype('uint8'))
        Idsave.save(tmp_name+'_color.png')
        #cv2.imwrite(tmp_name+'_color.png', colors)

        pred_map = np.zeros_like(pred_prob)
        for k, v in self.transLabel.items():
            pred_map[pred_prob==k] = v
        cv2.imwrite(tmp_name+'_labelId.png', pred_map)


def to_test(args):
    test_net = TestNet(args)
    test_net.load_net(args)
    for i in range(args.topK):
        time_now = time.time()
        image, image_name = test_net.load_image(args)
        for scale_i, scale in enumerate(test_net.aug_scale): #每种scale
            pred_list = [] ##预测的数据
            one_list, image_name, index = test_net.transfer_img(image, image_name, scale, args)
            for isave, im in enumerate(one_list): #剪成多张图片 一张一张喂进去
                test_net.net.blobs['data'].data[...] = im
                test_net.net.forward()
                pred_list.append(test_net.net.blobs['seg_pred'].data[0].copy())
                #cv2.imwrite('pred_{}.png'.format(isave), np.argmax(pred_list[isave],axis=0))
            test_net.save_pred(pred_list, image_name, [scale, scale_i], index, args) #之后将图片合起来
            pred_list = []
            one_list = []
        test_net.save_multi_results(image_name,  args)
        cost_time = time.time() - time_now
        print ('{} cost {}, remain time:{}'.format(image_name, cost_time, (args.topK-1)*cost_time))
if __name__ == '__main__':
    args = argument()
    to_test(args)
