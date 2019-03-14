#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''tar批量解压,path是tar存档目录
    $python3 tar_path.py path
'''

import os
import sys

#参数个数
argc = len(sys.argv);
if argc!=3:
    print('uage:%s from_path to_path'%(sys.argv[0]))
else:
    path=sys.argv[1]
    to_path=sys.argv[2]
    allfilelist=os.listdir(path);
    for file in allfilelist:
        #分离扩展名：os.path.splitext()
        vals=os.path.splitext(file) #=>('/home/temp/file','.txt')
        sname=vals[0]
        sextname=vals[1]
        tar_file=os.path.join(path,file)
        xpath=os.path.join(to_path,sname)
        if sextname=='.tar':
            if not os.path.exists(xpath):
                os.makedirs(xpath)
            cmd="tar -xvf %s -C %s"%(tar_file,xpath)
            print(cmd)
            os.system(cmd)

