#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''图像分割:分水岭算法
   https://blog.csdn.net/llh_1178/article/details/73321075
   https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_watershed/py_watershed.html?highlight=connectedcomponents

功能说明:
   分水岭算法分割图像,递归遍历目录,在保存目录以相对路径方式保存分割结果.
   一个源文件分割为多个子图时的新文件命名规则: 原文件_分割序号.扩展名,如:
      src.jpg分割了两个子图,则分别命名为 src_0.jpg,src_1.jpg
参数说明:
   --srcpath   :源文件目录
   --savepath  :分割结果保存目录
   --picktype  :子图提取模式
                0-掩码模式,提取掩码内容,其他部分以bgcolor填充
                1-外框模式,根据上下左右边界提取
   --bordersize:在子图大小基础上扩展边界大小
   --bgcolor   :背景颜色,picktype=0时有效
使用范例:
   $python3 img_seg.1.py --srcpath ./temp1 --savepath ./temp2 --picktype 0 --bordersize 0 --bgcolor '0,0,0'

'''

import os
import sys
import argparse
import pdb

import cv2
import numpy as np
#import matplotlib.pyplot as plt
dbg=0

def img_seg(img):
    '''图像分割
    @param img:图像数据
    @return markers:分割后的标签结果
    '''
    #转灰度图
    gray = img if img.ndim==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将颜色转变为灰色之后，可为图像设一个阈值，将图像二值化。
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # 下面用morphologyEx变换来除去噪声数据，这是一种对图像进行膨胀之后再进行腐蚀的操作，它可以提取图像特征：
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations= 2)
    # 通过对morphologyEx变换之后的图像进行膨胀操作，可以得到大部分都是背景的区域：
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # 接着通过distanceTransform来获取确定前景区域，原理是应用一个阈值来决定哪些区域是前景，越是远离背景区域的边界的点越可能属于前景。
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # 考虑前景和背景中有重合的部分，通过sure_fg和sure_bg的集合相减得到。
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg) #粘连区域未定区域,分割对象连在一起难以区分边界
    # 现在有了这些区域，就可以设定“栅栏”来阻止水汇聚了，这通过connectedComponents函数来完成
    ret, markers = cv2.connectedComponents(sure_fg) #通过联通性检测,把前景打上标签序号,分割序号以1,2,3,...排序分割区域
    # 在背景区域上加1， 将unknown区域设置为0：
    markers = markers + 1
    markers[unknown==255] = 0
    # 最后打开门，让水漫起来并把栅栏绘成红色:m2=cv2.watershed(img,m1) #m1,m2指向同一内存块
    markers = cv2.watershed(img, markers) #255即为分割的边界线
    
    if dbg==1:
        img_seg=img.copy()
        img_seg[markers == -1] = [255, 0, 0]
        cv2.imshow('src',img)                       #debug-原始图
        cv2.imshow('gray',gray)                     #debug-灰度图
        cv2.imshow('thresh',thresh)                 #debug-二值图
        cv2.imshow('open',opening)                  #debug-形态学-开操作
        cv2.imshow('sure_bg',sure_bg)               #debug-背景图
        cv2.imshow('dist_transform',dist_transform) #debug-开操作距离变换
        cv2.imshow('sure_fg',sure_fg)               #debug-前景图
        cv2.imshow('unknown',unknown)               #debug-(背景-前景)
        cv2.imshow('img',img_seg)                   #debug-分水岭分割结果
        cv2.imshow('markers',np.uint8(markers))     #debug-边界
        print('obj count:',markers.max())
        #print(ret)
        #for i in range(0,10):
        #    temp=np.zeros(img.shape,img.dtype)
        #    temp[markers==i]=img[markers==i]
        #    cv2.imshow('img_%d'%(i),temp)
        cv2.waitKey()
    return markers


def img_seg_pick(img,markers,ntype=0,bordersize=0,bgcolor=(0,0,0)):
    '''图像分割提取
    @param img:图像原始数据
    @param markers:分割掩码标签
    @param ntype:提取方式,0-掩码提取,1-外框提取
    @param bordersize:扩展边框大小
    @param bgcolor:背景颜色,ntype=0时有效
    @return img_list:提取对象
    '''
    img_list=[]
    img_count=markers.max()  #分割个数
    ndim=img.ndim
    for i in range(1,img_count+1):
        #分割子图索引
        index=np.array(np.where(markers==i)).T
        #子图左右上下边界
        top=index[:,0].min()
        bottom=index[:,0].max()
        left=index[:,1].min()
        right=index[:,1].max()
        #扩展边框修正
        top=top-bordersize if bordersize>0 and top-bordersize>0 else top
        bottom=bottom+bordersize if bordersize>0 and bottom+bordersize<=markers.shape[0] else bottom
        left=left-bordersize if bordersize>0 and left-bordersize>0 else left
        right=right+bordersize if bordersize>0 and right+bordersize<=markers.shape[1] else right
        if ntype==0:
            #掩码提取
            img_sub=np.ones(img.shape,img.dtype)
            if ndim==3:
                img_sub[:,:,0]=bgcolor[0]
                img_sub[:,:,1]=bgcolor[1]
                img_sub[:,:,2]=bgcolor[2]
            else:
                img_sub=img_sub*bgcolor[0]
            img_sub[markers==i]=img[markers==i]
            if ndim==3:
                img_sub=img_sub[top:bottom+1,left:right+1,:]
            else:
                img_sub=img_sub[top:bottom+1,left:right+1]
            img_list.append(img_sub)
        elif ntype==1:
            #外框提取
            if ndim==3:
                img_sub=img[top:bottom+1,left:right+1,:]
            else:
                img_sub=img[top:bottom+1,left:right+1]
            img_list.append(img_sub)
    return img_list


def imgfile_seg(sfile,savepath,picktype=0,bordersize=0,bgcolor=(0,0,0)):
    '''图像分割
    @param sfile:图像文件路径
    @param savepath:保存目录
    @param picktype:子图提取方式
    @param bordersize:边界扩展大小
    @param bgcolor:背景颜色
    @return img_list:分割后的图像列表
    '''
    if not os.path.exists(savepath):
        os.makedirs(savepath)    
    sname=os.path.splitext(os.path.basename(sfile))
    img=cv2.imread(sfile)
    if not img is None:
        markers=img_seg(img)
        img_list=img_seg_pick(img,markers,picktype,bordersize,bgcolor)
        for i,img_sub in enumerate(img_list):
            new_file='%s/%s_%d%s'%(savepath,sname[0],i,sname[1])
            cv2.imwrite(new_file,img_sub)

def imgs_seg(srcpath,dstpath):
    '''图像分割
       分割图像结果保存文件名规则:
       srcname_{i}.ext
         srcname --- 原文件名称 
         i --------- 分割序号
         ext ------- 原文件扩展名
    @param srcpath:图像源目录
    @param dstpath:分割后图像存放目录
    '''
    pass

def TravePath(src_path,src_root,dst_root,picktype=0,bordersize=0,bgcolor=(0,0,0)):
    allfilelist=os.listdir(src_path);
    for file in allfilelist:
        filepath=os.path.join(src_path,file);
        #判断是不是文件夹
        if os.path.isdir(filepath):
            TravePath(filepath,src_root,dst_root)
        else:
            print(filepath)
            #保存目录
            savepath='%s%s'%(dst_root,src_path[len(src_root):])
            #图像分割
            imgfile_seg(file,savepath,picktype,bordersize,bgcolor)


def params():
    ''' 程序参数
    '''
    #程序描述
    description='百度图像爬取'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--srcpath' ,type=str, help='源路径.   eg. --srcpath  "./temp1"',default='./temp1');
    parser.add_argument('--savepath',type=str, help='保存路径. eg. --savepath "./temp2"',default='./temp2');
    parser.add_argument('--picktype',type=int, help='提取模式,0-掩码,1-外框. eg. --picktype 0',default=0);
    parser.add_argument('--bordersize',type=int, help='扩展边框大小. eg. --bordersize 0',default=0);
    parser.add_argument('--bgcolor',type=str, help='背景颜色. eg. --bgcolor "0,0,0"',default='0,0,0');
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    #imgs_seg(arg.srcpath,arg.savepath)
    #imgfile_seg('1.jpg')
    #imgfile_seg('2.png')
    bgcolor=np.uint8(arg.bgcolor.split(','))
    TravePath(arg.srcpath,arg.srcpath,arg.savepath,arg.picktype,arg.bordersize,bgcolor)

if __name__=='__main__':
    arg=params()
    main(arg)

