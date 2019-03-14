#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''图像拷贝
'''

import cv2
import numpy as np

sfile='1.jpg'
img=cv2.imread(sfile)
mask=np.zeros(img.shape)
mask[20:100,20:100]=1

img2=img[20:100,20:100,:]

cv2.imshow('img1',img)
cv2.imshow('img2',img2)
cv2.waitKey()

aa=['a','b','c']
for i,s in enumerate(aa):
    print('%d:%s'%(i,s))
