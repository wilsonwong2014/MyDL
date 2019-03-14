#!/usr/bin/env python3
'''
## 构造训练样本
* 图像大小:224x224
* 单类样本大小:5000,train-2000,valid-1000,test-1000
* 样本类别：直线，多边形，圆，椭圆
* 生成数据目录结构
    /to/path/train/line
    /to/path/train/circle
    /to/path/train/ellipse
    /to/path/train/poly3
    /to/path/train/poly4
    /to/path/train/poly5
    /to/path/train/poly6
    /to/path/train/poly7
    /to/path/train/poly8
    /to/path/train/poly9
    ---------------------
    /to/path/valid/line
    /to/path/valid/circle
    /to/path/valid/ellipse
    /to/path/valid/poly3
    /to/path/valid/poly4
    /to/path/valid/poly5
    /to/path/valid/poly6
    /to/path/valid/poly7
    /to/path/valid/poly8
    /to/path/valid/poly9
    --------------------
    /to/path/test/line
    /to/path/test/circle
    /to/path/test/ellipse
    /to/path/test/poly3
    /to/path/test/poly4
    /to/path/test/poly5
    /to/path/test/poly6
    /to/path/test/poly7
    /to/path/test/poly8
    /to/path/test/poly9
'''
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mylibs.data.gen_gemotry import *

def gen_data(to_path,imgsize=(224,224,3),samples={'train':2000,'valid':1000,'test':1000},poly_N=(3,10),fill=False):
    '''构造测试样本数据
    @param to_path 输出目录
    @param imgsize 图像大小
    @param samples 构造样本大小
    @param poly_N  多边形的边数
    @param fill    闭合图形是否填充
    '''
    #to_path :输出目录
    if os.path.exists(to_path):
        print('%s already exists! please del first!!!'%(to_path))
    else:
        print('Generating data ......')
        #构造样本
        for k,v in samples.items():
            #构造直线样本
            print('Generating line data ......')
            tmp_path='%s/%s/line'%(to_path,k)
            os.makedirs(tmp_path) if not os.path.exists(tmp_path) else ''
            for i in range(v):
                sfile='%s/%s_line_%d.jpg'%(tmp_path,k,i)
                img=gen_line(imgsize)
                cv2.imwrite(sfile,img)
            print('Generating line data finished!')

            #构造圆形样本
            print('Generating circle data ......')
            tmp_path='%s/%s/circle'%(to_path,k)
            os.makedirs(tmp_path) if not os.path.exists(tmp_path) else ''
            for i in range(v):
                sfile='%s/%s_circle_%d.jpg'%(tmp_path,k,i)
                img=gen_circle(imgsize,fill=fill)
                cv2.imwrite(sfile,img)    
            print('Generating circle data finished!')

            #构造椭圆样本
            print('Generating ellipse data ......')
            tmp_path='%s/%s/ellipse'%(to_path,k)
            os.makedirs(tmp_path) if not os.path.exists(tmp_path) else ''
            for i in range(v):
                sfile='%s/%s_ellipse_%d.jpg'%(tmp_path,k,i)
                img=gen_ellipse(imgsize,fill=fill)
                cv2.imwrite(sfile,img)
            print('Generating ellilpse data finished!')

            #构造多边形样本
            print('Generating poly data ......')
            for edges in range(poly_N[0],poly_N[1]):
                tmp_path='%s/%s/poly%d'%(to_path,k,edges)
                os.makedirs(tmp_path) if not os.path.exists(tmp_path) else ''
                for i in range(v):
                    sfile='%s/%s_poly%d_%d.jpg'%(tmp_path,k,edges,i)
                    img=gen_poly(imgsize,edges,fill=fill)
                    cv2.imwrite(sfile,img)
            print('Generating poly data finished!')
        
        print('to_path:',to_path)
        print('Generating data finished!')

#==========================
if __name__=='__main__':
    import shutil

    to_path='%s/work/temp/gemotry'%os.getenv('HOME')
    if os.path.exists(to_path):
        shutil.rmtree(to_path)

    imgsize=(224,224,3)
    samples={'train':20,'valid':10,'test':10}
    poly_N=(3,10)
    fill=False
    gen_data(to_path,imgsize=imgsize,samples=samples,poly_N=poly_N,fill=fill)


