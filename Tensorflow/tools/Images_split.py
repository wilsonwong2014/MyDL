#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''图像分类
    如把图像分为训练、校验、测试
   源目录结构：
    顶级目录
    |---分类1子目录
        |---分类1文件1
        |---分类1文件2
        |---.........
    |---分类2子目录
    |---..........
   目的目录与源目录具有相同结构
   使用范例：
    $python3 Images_split.py --src srcpath --dst dstpath --split_class 'train,valid,test' --split_per '0.6,0.2,0.2'

'''

import os
import sys
import shutil
import pdb       
import numpy as np
import cv2
import argparse
from mylibs import funs

def params():
    ''' 程序参数
    '''
    #程序描述
    description='把源目录图像按比例分成指定分类'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--src',type=str, help='源目录.   eg. --src "./srcpath"',default='');
    parser.add_argument('--dst',type=str, help='目的目录. eg. --src "./dstpath"',default='');
    parser.add_argument('--split_class',type=str, help='分类列表. eg. --split_class "train,valid,test"',default='train,valid,test');
    parser.add_argument('--split_per',type=str, help='分类比例. eg. --split_per "0.6,0.2,0.2"',default='0.6,0.2,0.2');
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    funs.images_split(arg.src,arg.dst,arg.split_class,arg.split_per)


if __name__=='__main__':
    arg=params()
    if arg.src=='' or arg.dst=='' or arg.split_class=='' or arg.split_per=='':
        print('usge:%s --src srcpath --dst dstpath --split_class "train,valid,test" --split_per "0.6,0.2,0.2"')
    else:
        main(arg)

