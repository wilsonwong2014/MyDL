#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''图像文件读取性能测试
    读取文件前后各50个字节，并做md5计算
'''

import os
import sys
import time
import hashlib

path=sys.argv[1]
files=os.listdir(path)
t1=time.time()
m2=hashlib.md5()
ncount=0
for sfile in files:
    sfile=os.path.join(path,sfile)
    if not os.path.isfile(sfile):
        continue
    with(open(sfile,'rb')) as f:
        #读取前面50个字节
        bytes=f.read(50)
        #读取后面50个字节
        f.seek(50,2)#倒数50个字节
        bytes+=f.read(50)
        #md5
        m2.update(bytes)
        print(m2.hexdigest())
        ncount+=1
t2=time.time()

#---------------------------------------------
print('escape:%s,avg:%s',t2-t1,(t2-t1)/ncount)    
