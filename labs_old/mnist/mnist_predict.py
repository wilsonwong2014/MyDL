#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

模块功能:
    MNIST预测
    
参数说明:
    --modelfile    模型参数文件,默认""
    --testpath     预测图像目录,默认"~/data/mnist_mlp/test"
    --savefile     预测结果文件,默认"~/data/mnist_mlp/test/result.txt"
                       文件名:结果

使用范例:
    $python3 mnist_predict.py --modelfile ~/data/mnist_mlp/model/model.h5 --testpath ~/data/mnist_mlp/test --savefile ~/data/mnist_mlp/test/result.txt

'''
from __future__ import print_function

import os
import sys
import pdb

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import random
import argparse
import cv2

#-------------------------------
def predict(model,x):
    ''' 测试
    @param model:模型
    @param x:测试数据
    @return y:测试结果
    '''
    return model.predict(x)


#-------------------------------
def get_img(sfile):
    '''图像读取并预处理:28x28
    @return img:返回[28x28,1]行向量
    '''
    img=cv2.imread(sfile,cv2.IMREAD_GRAYSCALE) #读取图像
    if isinstance(img,np.ndarray):
        #图像预处理 ...
        img.resize((28,28))                        #重置图像大小
        return img.reshape((1,28*28))              #返回行向量
    else:
        return None


#-------------------------------
def predict_files(arg):
    '''模型预测,每一个文件都是一个单独的数字
       加载模型文件=>遍历预测目录，读取图形文件=>图像预测=>存放预测结果
    '''
    model_file=arg.modelfile #模型参数文件
    testpath=arg.testpath    #测试目录，不支持递归遍历
    savefile=arg.savefile    #测试结果保存文件
    if not os.path.exists(model_file):
        print('model_file:%s not exists!' %(model_file))
        return
    if not os.path.exists(testpath):
        print('testpath:%s not exists!' %(testpath))
        return
    #加载模型
    model = load_model(model_file)
    print('model.get_layer(0)======>')
    print(model.layers[0].input.shape[1:])
    #检索图像文件列表
    files=os.listdir(testpath)
    with open(savefile,'w') as f:
        for s in files:
            sfile='%s/%s' %(testpath,s)
            print('file:',sfile)
            x=get_img(sfile)
            #if isinstance(x.reshape(model.layers[0].input.shape(1:),np.ndarray):
            if isinstance(x,np.ndarray):
                #y=predict(model,x.reshape([1,model.layers[0].input.shape[1:]]))
                y=predict(model,x.reshape((1,28,28,1)))
                y_index=np.where(y==y.max())
                f.write('%s:%d:%s\r\n'%(s,y_index[1][0],y))
            else:
                print('get_img(%s):=> fail!'%(s))


#-------------------------------
def params():
    ''' 程序参数
    '''
    #程序描述
    description='MNIST预测'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--modelfile', type=str, help='模型参数文件. eg. --modelfile "./temp/temp.h5"', default='');
    parser.add_argument('--testpath', type=str, help='预测图像目录. eg. --testpath "~/data/mnist_mlp/test"', default='%s/data/mnist_mlp/test'%(os.getenv('HOME')));
    parser.add_argument('--savefile', type=str, help='预测结果. eg. --savefile "~/data/mnist_mlp/test/result.txt"', default='%s/data/mnist_mlp/test/result.txt'%(os.getenv('HOME')));
    parser.add_argument('--dbg', type=int, help='是否调试. eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    #--------------------------
    return arg


#-------------------------------
def main(arg):
    ''' 主函数
    '''
    predict_files(arg)

#==============================
if __name__=='__main__':
    arg=params() #命令行参数解析
    print('arguments:')
    print('--modelfile:',arg.modelfile)
    print('--testpath:',arg.testpath)
    print('--savefile:',arg.savefile)
    print('--dbg:',arg.dbg)
    print('--------------end arguments-----------------')
    main(arg)

