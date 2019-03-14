222222222#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

'''
cirfar10数据集输出图像文件
模块功能:
    输出cirfar10数据集到图像文件

参数说明:
    --savepath     保存路径,默认 ./temp
    --max_trains   训练最大输出文件数,0表示所有,默认0
    --max_tests    测试最大输出文件数,0表示所有,默认0
使用范例:
    $python3 cirfar10_output_img.py --savepath ~/data/cirfar10/test1 --rows 30 --cols 30 --max_trains 0 --max_tests 0
    
    $python3 cirfar10_output_img.py --savepath ~/data/cirfar10/test2 --rows 1 --cols 1 --max_trains 100 --max_tests 100
'''

import os
import sys
import pdb       
#import time
from keras.datasets import cifar10
#import struct
import numpy as np
import cv2
import argparse


def save_to_img(to_path,x,y,max_num=0):
    ''' 把数据保存到图像组--使用opencv方法.
    @param to_path:   保存路径.
    @param x:         图像数据 [n ,32,32,3]
    @param y:         标签数据 [n x 1]
    @param max_num:   最大输出文件数,0表示所有,默认0
    '''
    #目录创建
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    #创建输出图像
    image_num=x.shape[0] if max_num==0 or x.shape[0]<max_num else max_num
    index=0
    for i in range(image_num):
        filepath="%s/%d_%d.png" %(to_path,i,y[i])
        print(filepath)
        cv2.imwrite(filepath,x[i,:,:,:])


def params():
    ''' 程序参数
    '''
    #程序描述
    description='cifar10数据集输出图像文件'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--savepath',type=str
        , help='保存路径. eg. --savepath "%s/data/cifar10"'%(os.getenv('HOME')),default='%s/data/cifar10'%(os.getenv('HOME')));
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
    (x_train,y_train),(x_test,y_test)=cifar10.load_data();
    to_path='%s/train' %(arg.savepath)
    save_to_img(to_path,x_train,y_train,arg.max_trains);
    to_path='%s/test' %(arg.savepath)
    save_to_img(to_path,x_test,y_test,arg.max_tests);


if __name__=='__main__':
    arg=params()
    main(arg)

