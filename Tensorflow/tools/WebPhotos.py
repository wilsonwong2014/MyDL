#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''自动构建图像网站
'''
import os
import sys
import pdb
import argparse
import cv2
from mylibs import funs
from mylibs import ProcessBar

#创建缩略图
def CreateThumb_img(src_file,dst_file,width=100,height=100):
    src=cv2.imread(src_file)                                        #读取源图像
    if not src is None:
        dst=cv2.resize(src,(width,height))                              #修改图像尺寸
        dst_path=os.path.split(dst_file)[0]                             #目录检测
        os.makedirs(dst_path) if not os.path.exists(dst_path) else ''   #目录检测
        cv2.imwrite(dst_file,dst)                                       #保存缩略图

#创建缩略图(视频)
def CreateThumb_vedio(src_file,dst_file,width=100,height=100):
    cap = cv2.VideoCapture(src_file)                                    #打开视频流 
    if cap.isOpened():
        success, frame = cap.read()                                     #读取第一贞
        if not frame is None:
            dst=cv2.resize(frame,(width,height))                            #修改图像尺寸
            dst_path=os.path.split(dst_file)[0]                             #目录检测
            os.makedirs(dst_path) if not os.path.exists(dst_path) else ''   #目录检测
            cv2.imwrite(dst_file,dst)                                       #保存缩略图
    else:
        print('%s can not open!'%(src_file))
    cap.release() 


#创建缩略图
def CreateThumbs(path,ori_name,thumb_name,exts='',exclude_files='',exclude_exts='',width=100,height=100):
    '''创建缩略图
    @param path 相册根目录
    @param ori_name 原始图像目录名称
    @param thumb_name 缩略图像目录名称
    @param width 缩略图宽度
    @param height 缩略图高度
    '''
    ori_path  ='%s/%s'%(path,ori_name)  #原始图像路径
    thumb_path='%s/%s'%(path,thumb_name)#缩略图像路径
    if not os.path.exists(ori_path):
        print('%s not exists!'%(ori_path))
    else:
        #搜集原始文件列表
        ori_files=[]
        funs.GatherFilesEx(ori_path,ori_files,exts=exts,exclude_files=exclude_files,exclude_exts=exclude_exts)
        files_num=len(ori_files)
        #支持视频格式
        vedio_exts='.mp4'       
        #创建缩略图
        pbar=ProcessBar.ShowProcess()
        rep_ori_name='/%s/'%(ori_name)      #原始目录标记
        rep_thumb_name='/%s/'%(thumb_name)  #缩略目录标记
        for i,sfile in enumerate(ori_files):
            src_file=sfile
            dst_file=src_file.replace(rep_ori_name,rep_thumb_name)
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                #print('src:%s'%(src_file))
                #print('dst:%s'%(dst_file))
                if os.path.splitext(src_file)[1] in vedio_exts:
                    #视频
                    CreateThumb_vedio(src_file,'%s.jpg'%(dst_file),width,height)
                else:
                    #图像
                    CreateThumb_img(src_file,dst_file,width,height)
            if i%10==0:
                pbar.show_process(int(i*100/files_num))
        pbar.show_process(100)  

#构建html页面
def CreateHtml(root_path,path,ori_name,thumb_name,exclude_files='',exclude_exts='.htm,.html',width=100,height=100):
    '''
    <html>
    <head>
    </head>
    <body>
    path:[<a url='2017'>2017</a>][<a url='2018'>2018</a>]<p>
    cur_path:./aaa <p>
    Images:
    <img src='1.jpg' width=100 height100>1.jpg</img>&nbsp&nbsp
    <img src='2.jpg' width=100 height100>2.jpg</img>&nbsp&nbsp
    <img src='3.jpg' width=100 height100>3.jpg</img>&nbsp&nbsp
    <p>
    </body>
    </html>
    '''
    rep_ori_name='/%s/'%(ori_name)
    rep_thumb_name='/%s/'%(thumb_name)
    vedio_exts='.mp4'
    paths=''    #子目录序列
    cur_path='cur_path:%s &nbsp;<p>'%(path)
    if not os.path.exists(path):
        print('%s not exists!'%(path))
    else:    
        imgs=''
        items=os.listdir(path)
        for sfile in items:
            temp_path=os.path.join(path,sfile)
            if os.path.isdir(temp_path):
                print('当前处理目录：%s'%(temp_path))
                paths+='[<a href="%s/index.htm">%s</a>]&nbsp;'%(sfile,sfile)
                CreateHtml(root_path,temp_path,ori_name,thumb_name,exclude_files=exclude_files,exclude_exts=exclude_exts,width=width,height=height)
            else:
                 if sfile not in exclude_files and os.path.splitext(sfile)[1] not in exclude_exts:
                    surl=temp_path.replace(rep_thumb_name,rep_ori_name)[len(root_path):]
                    v_ext=os.path.splitext(os.path.splitext(sfile)[0])[1]
                    if (not v_ext=='') and v_ext in vedio_exts:
                        #视频:如 {./temp/1.mp4.jpg => ./temp/1.mp4}
                        surl=os.path.splitext(surl)[0]
                        imgs+='<a href="%s">[vedio]<img src="%s"/></a>&nbsp;\r\n'%(surl,sfile)
                    else:
                        #图像
                        imgs+='<a href="%s"><img src="%s"/></a>&nbsp;\r\n'%(surl,sfile)
                    
        shtml='<html>'\
              '<head>'\
              '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />'\
              '</head>'\
              '<body>'\
              'paths:%s<p>'\
              'cur_path:%s<p>'\
              '<a href="../index.htm">parent</a><p>'\
              '%s'\
              '</body>'\
              '</html>'%(paths,path[len(root_path):],imgs)
        
        filepath='%s/index.htm'%(path);
        #写方式打开文件
        with open(filepath,'w') as f:
            #写入内容 
            f.write(shtml);
            #关闭文件

def params():
    ''' 程序参数
    '''
    #程序描述
    description='自动构建相册网页'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--path',type=str, help='网站根目录. eg. --path "~/data/temp/pthtos"',default='/home/hjw/data/temp/test');
    parser.add_argument('--width', type=int, help='缩略图宽度. eg. --width 100', default=100);
    parser.add_argument('--height',type=int, help='缩略图高度. eg. --height 100 ',default=100);
    parser.add_argument('--exts',type=str, help='搜索扩展名列表. eg. --exts ".jpg,.jpeng,.png"',default='');
    parser.add_argument('--exclude_files',type=str, help='排除文件列表. eg. --exts "db.json,num.txt"',default='');
    parser.add_argument('--exclude_exts',type=str, help='排除扩展名列表. eg. --exts ".json,.txt"',default='.htm,.html');
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    path=arg.path
    CreateThumbs(arg.path,'Original','Thumb',exclude_files=arg.exclude_files,exclude_exts=arg.exclude_exts,width=arg.width,height=arg.height)
    CreateHtml(arg.path,'%s/Thumb'%(arg.path),'Original','Thumb',exclude_files=arg.exclude_files,exclude_exts=arg.exclude_exts,width=arg.width,height=arg.height)

if __name__=='__main__':
    arg=params()
    main(arg)

