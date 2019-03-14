#!/usr/bin/env  python3
# -*- coding: utf-8 -*-

'''图像显示
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

#显示多张图像
def show_images(images,grids=(16,16),grid_size=(150,150),margin=1,figsize=None,title=None):
    '''多张图像显示
        cv2.imread(img_file) #=> np.array(size=[height,width,channels])
    @param images    图像序列,RGB,dtype=uint8
                        np.array.shape:(batch_size,height,width,channels) 
                        或
                        list[np.array(size=(height,width,channels))]
    @param grids     图像网格数   (rows,cols)
    @param gird_size 网格图像大小 (width,height)
    @param marign    图像间隙
    @param figsize   figure尺寸,tuple,(width,height)
    '''
    import cv2
    import matplotlib.pyplot as plt
    
    #初始化图像网格[rows{height} x cols{width}]
    rows,cols=grids                       #图像网格数(rows,cols)
    width,height=grid_size                #图像网格大小(width,height)
    g_width= cols * width+(cols-1)*margin #画布宽度
    g_height=rows*height+(rows-1)*margin  #画布高度
    display_grid = np.zeros((g_height,g_width ,3),dtype=np.uint8)  #初始化图像画布
    
    row=0 #当前行
    col=0 #当前列
    for index,img in enumerate(images):
        if img is None:
            #非法图像处理
            continue
        if index>=rows*cols:
            #越界处理
            break;
        if(not np.issubdtype(img.dtype,np.uint8)):
            #图像数据类型处理：统一为uint8格式
            fmax=np.max(img)
            fmin=np.min(img)
            img=((img-fmin)/(fmax-fmin))*255
            img=img.astype(np.uint8)
        
        img=cv2.resize(img,(width,height)) #子图大小调整 
        
        #子图显示区域计算
        row=index // cols                  #当前行
        col=index - row*cols               #当前列
        row_start=row*height+row*margin
        row_end  =row_start+height
        col_start=col*width+col*margin
        col_end  =col_start+width
        #子图拷贝
        display_grid[row_start : row_end,col_start:col_end] = img        
    
    #画布显示
    if figsize!=None:
        dpi=72.0
        fig_width,fig_height=figsize
        if fig_width==None:
            fig_width=(fig_height*g_width*1.0/g_height)
        if fig_height==None:
            fig_height=(fig_width*g_height*1.0/g_width)
        plt.figure(figsize=(fig_width/dpi,    #width ,英寸
                            fig_height/dpi))  #height,英寸                
    else:
        plt.figure #系统默认方式
    plt.grid(True)
    plt.imshow(display_grid,aspect='auto')
    if title!=None:
        plt.title(title)
    
    plt.show()  

#==========================================
if __name__=='__main__':
    #测试：多张图像显示
    import os
    import cv2
    import matplotlib.pyplot as plt

    files=['%s/work/data/gtest/jpg10/%d.jpg'%(os.getenv('HOME'),i) for i in range(10)]
    images=[cv2.imread(sfile) for sfile in files]
    images=[cv2.cvtColor(img,cv2.COLOR_BGR2RGB) for img in images]
    show_images(images,grids=(3,3),grid_size=(100,50),figsize=(1204,300))
