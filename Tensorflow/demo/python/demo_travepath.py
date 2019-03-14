#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################
#   文件夹遍历范例      #
########################

import os
import sys;
def TravePath(path):
    allfilelist=os.listdir(path);
    for file in allfilelist:
        filepath=os.path.join(path,file);
        #判断是不是文件夹
        if os.path.isdir(filepath):
            TravePath(filepath);
        else:
            print(filepath);
            #获取文件名(不含扩展名)
            index = filepath.rfind('.');  
            filepath=filepath[:index];
            sPyFile = filepath+".py";
            print(sPyFile);  
            if(not os.path.exists(sPyFile)):
                #os.system("ls %s" %filepath.replace(" ","\\ "));
                os.system("jupyter nbconvert --to script %s" %filepath.replace(" ","\\ "));


if __name__ == '__main__':
    if len(sys.argv)==2:
        TravePath(sys.argv[1]);
    else:
        print("Usage: demo_travpath.py /path");


