#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''获取图像类型
'''
import os
import imghdr
sfile='%s/work/data/1.jpg'%(os.getenv('HOME'))
print('sfile:',sfile)
stype=imghdr.what(sfile)
print(stype) #=>jpeg
