#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'First string line is comment!'

'''根据指定扩展名搜集文件列表
    使用范例：
        $python3 GatherFiles.py ./temp ".jpg,.png"
'''



__author__ = 'wilsonwong'

import os 
import sys

from  mylibs import funs


#------------------------
if __name__=='__main__':
    if len(sys.argv)<2:
        print('usge:%s /path ".jpg,.png"'%(sys.argv[0]))
        sys.exit()
    path=sys.argv[1]
    exts=''
    if len(sys.argv)>2:
        exts=sys.argv[2]
    files=funs.GatherFiles(path,exts=exts)
    print('GatherFiles:')
    for sfile in files:
        print(sfile)
    print('-----------')
    files=[]
    funs.GatherFilesEx(path,files,exts=exts)
    print('GatherFilesEx:')
    for sfile in files:
        print(sfile)
