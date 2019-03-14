#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'First string line is comment!'

'''通用函数集
'''

__author__ = 'wilsonwong'

import os 
import sys
import shutil
import imghdr
import cv2
from PIL import Image
from mylibs import ProcessBar

#------------文件搜集--------------
def GatherFiles(path,**kwargs):
    '''文件搜集
    @param path 搜索目录,递归遍历
    @param exts 搜索文件扩展名,如:".jpg,.jpeg,.bmp,.png"
    @param exclude_files  排除文件列表，如："db.json,num.txt"
    @param exclude_exts   排除扩展名列表，如:".json,.txt"
    @return 搜索结果 list
        范例：
        ['/temp/1/a.jpg','/temp/2/b.jpg']
    
    Example:
        files=GatherFiles("./temp",exts=".jpg,.png",exclude_files="db.json,num.txt",exclude_exts=".json,.txt")
        for x in files:
            print(x)
    '''
    #-----------参数处理-----------
    exts=''
    exclude_files=''
    exclude_exts=''
    for k,v in kwargs.items():
        if k=='exts':
            exts=v
        elif k=='exclude_files':
            exclude_files=v
        elif k=='exclude_exts':
            exclude_exts=v
    #--------------------------
    files=[]
    items=os.listdir(path)
    for sfile in items:
        temp_path=os.path.join(path,sfile)
        if os.path.isdir(temp_path):
            files.extend(GatherFiles(temp_path,exts=exts,exclude_files=exclude_files,exclude_exts=exclude_exts))  #合并
        else:
            ext_name=os.path.splitext(temp_path)[1].lower()
            if exts=='' or ext_name in exts:
                #扩展名匹配
                if exclude_files=='' or not sfile in exclude_files:
                    #排除文件
                    if exclude_exts=='' or not ext_name in exclude_exts:
                        #排除扩展名
                        files.append(temp_path) #追加
    #end{for sfile in items:}
    return files
#==========================================
def GatherFilesEx(path,files,**kwargs):
    '''文件搜集
    @param path 搜索目录,递归遍历
    @param files 返回搜索结果
    @param exts 搜索文件扩展名,如:".jpg,.jpeg,.bmp,.png"
    @param exclude_files  排除文件列表，如："db.json,num.txt"
    @param exclude_exts   排除扩展名列表，如:".json,.txt"
    @return 搜索结果 list
        范例：
        ['/temp/1/a.jpg','/temp/2/b.jpg']
    
    Example:
        files=[]
        GatherFilesEx("./temp",files,exts=".jpg,.png",exclude_files="db.json,num.txt",exclude_exts=".json,.txt")
        for x in files:
            print(x)
    '''
    #-----------参数处理-----------
    exts=''
    exclude_files=''
    exclude_exts=''
    for k,v in kwargs.items():
        if k=='exts':
            exts=v
        elif k=='exclude_files':
            exclude_files=v
        elif k=='exclude_exts':
            exclude_exts=v
    #--------------------------
    #files=[]
    items=os.listdir(path)
    for sfile in items:
        temp_path=os.path.join(path,sfile)
        if os.path.isdir(temp_path):
            GatherFilesEx(temp_path,files,exts=exts,exclude_files=exclude_files,exclude_exts=exclude_exts)
        else:
            ext_name=os.path.splitext(temp_path)[1].lower()
            if exts=='' or ext_name in exts:
                #扩展名匹配
                if exclude_files=='' or not sfile in exclude_files:
                    #排除文件
                    if exclude_exts=='' or not ext_name in exclude_exts:
                        #排除扩展名
                        files.append(temp_path) #追加
    #end{for sfile in items:}
    return len(files)


#--------------文件夹统计信息---------------
def PathStat(path):
    '''统计文件夹信息：目录个数，文件个数，大小{Byte}
    @param path ---统计目录
    @return list
        如：
            [5,10,1024]
    使用范例：
        info=PathStat('/temp')
    '''
    info=[0,0,0]
    files=os.listdir(path)
    for sfile in files:
        file_path=os.path.join(path,sfile)
        if os.path.isdir(file_path):
            info_sub=PathStat(file_path) #遍历子目录
            info[0]+=info_sub[0] #子目录统计
            info[1]+=info_sub[1] #文件个数统计   
            info[2]+=info_sub[2] #文件大小统计
            info[0]+=1           #子目录加1
        else:
            info[1]+=1                          #文件个数统计
            info[2]+=os.stat(file_path).st_size #文件大小统计
    return info

#-------------gif转png------------------
def gif2png(sfile,to_path):
    '''gif转jpg或png
    https://blog.csdn.net/huxiangen/article/details/80825181 
    @param sfile    --- gif文件
    @param to_path  --- 保存路径
    使用范例：
        gif2png('a.gif','./temp')
        =>
        a_0.png
        a_1.png
        ......
    '''
    def iter_frames(im): 
        try: 
            i= 0 
            while 1: 
                im.seek(i) 
                imframe = im.copy() 
                if i == 0: 
                    palette = imframe.getpalette() 
                else: 
                    imframe.putpalette(palette) 
                yield imframe 
                i += 1 
        except EOFError: 
            pass
    im=Image.open(sfile)
    if not im is None:
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        file_sname=os.path.splitext(os.path.split(sfile)[1])[0]
        for i, frame in enumerate(iter_frames(im)): 
            frame.save('%s/%s_%d.png'%(to_path,file_sname,i),**frame.info)
        
#-----查找扩展名与类型不匹配的图像文件-----
def FindExtNotMatchFiles(path,exts,bFix=False):
    '''由于imghdr.what(sfile)返回的类型为空时，图像文件仍有效！本函数的比较作用失去意义！
    建议：停用！
    '''
    print('FindExtNotMatchFiles(%s,exts=%s,bFix=%d)'%(path,exts,bFix))
    rets=[]
    print('GatherFiles(%s,exts=%s)'%(path,exts))
    files=GatherFiles(path,exts)
    nFiles=len(files)
    pb = ProcessBar.ShowProcess(100,'Not Match Search','', 'OK') 
    #不匹配查询
    for i, sfile in enumerate(files):
        ext1=os.path.splitext(sfile)[1]
        ext2=imghdr.what(sfile)
        ext2='.' if ext2==None else '.'+ext2
        if ext1!=ext2 and not (ext1=='.jpg' and ext2=='.jpeg'):
            rets.append((sfile,ext2)) 
            if bFix:
                new_file='%s%s'%(os.path.splitext(sfile)[0],ext2)
                shutil.move(sfile,new_file)
        if i%50==0:
            pb.show_process(int(i*100/nFiles))
    pb.show_process(100)
    return rets


#----查找GIF文件----
def FindGIFFiles(path,delflag=0,unzipflag=0):
    '''查找GIF文件
    @param path -------- 搜索目录
    @param delflag ----- 删除标记,0-保留GIF文件，1-删除搜索到的GIF文件
    @param unzipflag --- 图像提取标记，0-忽略操作，1-把GIF的图像序列提取到同目录下，文件命名为：sfile_{n}.png
    @return list
        如：['/temp/a.gif','/temp/b.gif']
        
    使用范例：
        files=FindGIFFIles('/temp',0,0)
    '''
    print('FindGIFFIles(%s,delflag=%d,unzipflag=%d)'%(path,delflag,unzipflag))
    gif_files=[]
    print('GatherFiles(%s,exts="")'%(path))
    files=GatherFiles(path)
    nFiles=len(files)
    pb = ProcessBar.ShowProcess(100,'GIF Search','', 'OK') 
    for i,sfile in enumerate(files):
        #gif文件处理
        if imghdr.what(sfile)=='gif':
            gif_files.append(sfile)
            if unzipflag==1:
                gif2png(sfile,os.path.split(sfile)[0])
            if delflag==1:
                os.remove(sfile)
        if i%50==0:
            pb.show_process(int(i*100/nFiles))
    pb.show_process(100)
    return gif_files


#---查找无效文件{判断标准，cv2无法打开}-----                
def FindInvalidImg(path,delflag=0,exclude_files='',exclude_exts=''):
    '''查找非法图像文件
    @param path ---- 搜索路径
    @param delflag --- 删除标记，0-忽略，1-删除满足条件的搜索文件
    @param exclude_files --- 排除文件列表，如："db.json,num.txt"
    @param exclude_exts ---- 排除扩展名列表，如:".json,.txt"
    @return list
        如：
        ['/temp/a.jpg','/temp/a/b.jpg']
    使用范例：
        files=FindInvalidImg('./temp',0,"db.json,num.txt",".json,.txt")
    '''
    print('FindInvalidImg(%s,delflag=%d)'%(path,delflag))
    invalid_files=[]
    print('GatherFiles(%s,exts='',exclude_files="%s",exclude_exts="%s")'%(path,exclude_files,exclude_exts))
    files=GatherFiles(path,exclude_files=exclude_files,exclude_exts=exclude_exts)
    nFiles=len(files)
    pb = ProcessBar.ShowProcess(100,'Inv Search','', 'OK') 
    print('Search results:')
    for i,sfile in enumerate(files):
        img=cv2.imread(sfile)
        if img is None:
            invalid_files.append(sfile)
            if delflag==1:
                os.remove(sfile)
        if i%50==0:
            pb.show_process(int(i*100/nFiles))
    pb.show_process(100)
    return invalid_files

#------图像分集--把图像分成若干子集-----------------------
def images_split(src,dst,split_class,split_per):
    '''图像分集--把图像分成若干子集
    @param src          源目录,每个子目录表示一个分类，如:
        ./src/dog/1.jpg
        ./src/dog/2.jpg
        ./src/cat/1.jpg
        ./src/cat/2.jpg
    @param dst          目的目录,根据split_class划分为几个数据集，如：
        ./dst/train/dog/
        ./dst/train/cat/
        ./dst/valid/dog/
        ./dst/valid/cat/
        ./dst/test/dog/
        ./dst/test/cat/
    @param split_class  分类列表，如:"train,valid,test"
    @param split_per    分类比例，如:"0.6,0.2,0.2"，如果>1则为个数，如："2000,1000,1000"
    范例：
        images_split("./src","./dst",split_class="train,valid,test",split_per="0.6,0.2,0.2")
        或
        images_split("./src","./dst",split_class="train,valid,test",split_per="100,20,20")
    '''
    src_path=src
    dst_path=dst
    #图像分类
    split_class=split_class.split(',')
    #图像分类比例
    split_per=[float(x) for x in split_per.split(',')]
    #图像分类个数
    num_class=len(split_class) if len(split_class)<len(split_per) else len(split_per)
    #检索图像文件列表
    path_top=os.listdir(src_path)
    for sub_dir in path_top:
        path_sub=os.path.join(src_path,sub_dir)
        if os.path.isdir(path_sub):
            #遍历分类子目录
            files=os.listdir(path_sub)
            num_file=len(files)
            #图像分类比例转个数
            split_per_num=[int(num_file*x) if x<1 else int(x)  for x in split_per[:num_class]]
            start_index=0
            for i,n in enumerate(split_per_num):
                file_dir=os.path.join(dst_path,split_class[i],sub_dir)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                end_index=start_index+n
                for sfile in files[start_index:end_index]:
                    file_name=os.path.split(sfile)[1]
                    src_file=os.path.join(path_sub,file_name)
                    dst_file=os.path.join(dst_path,split_class[i],file_dir,file_name)
                    #print('copy src:',src_file)
                    #print('to   dst:',dst_file)
                    shutil.copy(src_file, dst_file)
                #====================
                start_index=end_index



