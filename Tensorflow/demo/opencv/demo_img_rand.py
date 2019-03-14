#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''随机图像
'''

import numpy as np
import cv2

img1=np.random.rand(512,512,1) #随机生成0-1的浮点图像
img2=img1*100                  #图像值增大100倍,大于1的显示为白色(255)
img3=cv2.normalize(img2, 0, img2.max(), cv2.NORM_MINMAX); #归一化处理

cv2.imshow('src',img1)
print('img1.min()=>',img1.min())
print('img1.max()=>',img1.max())
cv2.imshow('rand',img2)
print('img2.min()=>',img2.min())
print('img2.max()=>',img2.max())
cv2.imshow('rand-normalize',img3)
print('img2.min()=>',img3.min())
print('img2.max()=>',img3.max())

cv2.waitKey()
