#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 递归遍历文件夹，修改文件/目录名称
'''

import os
import sys
import shutil

#参数个数
argc = len(sys.argv);
if argc!=4:
    print('参数个数不够，至少3个!')
    print('usage: %s path old new'%(sys.argv[0]))
    sys.exit()

path  =sys.argv[1] #遍历文件夹
oldstr=sys.argv[2] #旧字符
newstr=sys.argv[3] #新字符

def TravePath(path,oldstr,newstr):
    allfilelist=os.listdir(path);
    for file in allfilelist:
        filepath=os.path.join(path,file);
        print(filepath);
        #判断是不是文件夹
        if os.path.isdir(filepath):
            TravePath(filepath,oldstr,newstr);
            oldpath=filepath                        #旧目录
            newpath=oldpath.replace(oldstr, newstr) #新目录
            shutil.move(oldpath,newpath)            #移动文件或重命名
        else:
            oldfile=file                             #旧文件名
            newfile=oldfile.replace(oldstr, newstr)  #新文件名
            shutil.move(os.path.join(path,oldfile),os.path.join(path,newfile))  #移动文件或重命名


if __name__ == '__main__':
    if len(sys.argv)==4:
        TravePath(sys.argv[1],sys.argv[2],sys.argv[3]);
    else:
        print("Usage: rename_replace.py /path old new");




