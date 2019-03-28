222222222#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

'''
mnist数据集输出图像文件
   http://yann.lecun.com/exdb/mnist/

模块功能:
    输出mnist数据集到图像文件

参数说明:
    --savepath     保存路径,默认 ./temp
    --rows         图像文件单元行数,默认30
    --cols         图像文件单元列数,默认30
    --max_trains   训练最大输出文件数,0表示所有,默认0
    --max_tests    测试最大输出文件数,0表示所有,默认0
使用范例:
    $python3 mnist_output_img.py --savepath ~/Data/mnist/test1 --rows 30 --cols 30 --max_trains 0 --max_tests 0
    
    $python3 mnist_output_img.py --savepath ~/Data/mnist/test2 --rows 1 --cols 1 --max_trains 100 --max_tests 100
'''

import os
import sys
import pdb       
import time
from keras.datasets import mnist
import struct
import numpy as np
import cv2
import argparse

def load_data(files=()):
    ''' 加载mnist数据集

    @param files: turple类型,
                  None--直接调用keras.dataset.mnist.load_data,
                  其他--files(0)训练输入文件,files(1)训练输出文件,files(2)测试输入文件,files(3)测试输出文件
    @return: (x_train, y_train,x_test, y_test)
             x_train ---[n x 784],训练输入数据,每行表示一张图像(28x28)
             y_train ---[n x 1  ],训练标签数据,0-9
             x_test ----[n x 784],测试输入数据,每行表示一张图像(28x28)
             y_test ----[n x 1  ],测试标签数据,0-9
    '''
    if len(files)<4:
        #直接调用keras.dataset.mnist.load_data
        (x_train, y_train), (x_test, y_test) = mnist.load_data();
    else:
        ##文件解析
        #x_train
        with open(files(0),'rb') as f:
            (magic_number,number_of_images,number_of_rows,number_of_cols)=struct.unpack('>iiii',f.read(32*4));
            x_train = np.fromfile(f,dtype=np.uint8).reshape(number_of_images, number_of_rows*number_of_cols);
        #y_train
        with open(files(1),'rb') as f:
            (magic_number,number_of_images)=struct.unpack('>ii',f.read(32*2));
            y_train = np.fromfile(f,dtype=np.uint8).reshape(number_of_images, 1);
        #x_test
        with open(files(2),'rb') as f:
            (magic_number,number_of_images,number_of_rows,number_of_cols)=struct.unpack('>iiii',f.read(32*4));
            x_test = np.fromfile(f,dtype=np.uint8).reshape(number_of_images, number_of_rows*number_of_cols);
        #y_test
        with open(files(3),'rb') as f:
            (magic_number,number_of_images)=struct.unpack('>ii',f.read(32*2));
            y_test = np.fromfile(f,dtype=np.uint8).reshape(number_of_images, 1);
    return (x_train, y_train,x_test, y_test);


def save_to_img(to_path,x,y,rows=30,cols=30,max_num=0):
    ''' 把数据保存到图像组--使用opencv方法.
       每张图像由rows x cols个单元组成,每个单元保存一张图像.
    
    @param to_path:   保存路径.
    @param x:         图像数据 [n x 784],每行表示一张图像(28 x 28)
    @param y:         标签数据 [n x 1]  ,每行表示一个图像标签(0..9)
    @param rows:      图像组行数
    @param cols:      图像组列数
    @param max_num:   最大输出文件数,0表示所有,默认0
    '''
    #目录创建
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    #创建输出图像
    image=np.zeros((rows*28,cols*28),dtype=np.uint8)
    image_num=x.shape[0] if max_num==0 or x.shape[0]<max_num else max_num
    frame_index=0
    imgs_per_frame=rows*cols;
    index=0
    num=0
    while(index<image_num):
        #write
        for i in range(rows):
            for j in range(cols):
                num=y[index]
                image[i*28:(i+1)*28,j*28:(j+1)*28]=x[index,:].reshape(28,28)
                index+=1
                if index>=image_num:
                    break
            if index>=image_num:
                break
        #Output
        frame_index+=1 
        filepath="%s/%d_%d.png" %(to_path,frame_index,num)
        print(filepath)
        cv2.imwrite(filepath,image)


def params():
    ''' 程序参数
    '''
    #程序描述
    description='mnist数据集输出图像文件,每个图像单元大小(28x28),每个图像文件大小为(rows x cols)个单元.'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--savepath',type=str
        , help='保存路径. eg. --savepath "%s/Data/mnist"'%(os.getenv('HOME')),default='%s/Data/mnist'%(os.getenv('HOME')));
    parser.add_argument('--rows',type=int, help='图像文件单元行数. eg. --rows 30',default=30);
    parser.add_argument('--cols',type=int, help='图像文件单元列数. eg. --cols 30',default=30);
    parser.add_argument('--max_trains',type=int, help='训练最大输出文件数,0表示所有,默认0. eg. --max_trains 0',default=0);
    parser.add_argument('--max_tests',type=int, help='测试最大输出文件数,0表示所有,默认0. eg. --max_tests 0',default=0);
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    (x_train,y_train,x_test,y_test)=load_data();
    to_path='%s/train' %(arg.savepath)
    save_to_img(to_path,x_train,y_train,arg.rows,arg.cols,arg.max_trains);
    to_path='%s/test' %(arg.savepath)
    save_to_img(to_path,x_test,y_test,arg.rows,arg.cols,arg.max_tests);


if __name__=='__main__':
    arg=params()
    main(arg)

