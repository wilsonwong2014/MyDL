#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

'''距离变换
cv2.distanceTransform(src, distanceType, maskSize[, dst ])
Parameters
    src – 8-bit, single-channel (binary) source image.
    dst – Output image with calculated distances. It is a 32-bit floating-point,
          single-channel image of the same size as src .
    distanceType – Type of distance. It can be CV_DIST_L1, CV_DIST_L2 , or CV_DIST_C .
    maskSize – Size of the distance transform mask. It can be 3, 5, or CV_DIST_MASK_PRECISE
           (the latter option is only supported by the first function). In case of the CV_DIST_L1
           or CV_DIST_C distance type, the parameter is forced to 3 because a 3 x 3 mask gives 
           the same result as 5 x 5 or any larger aperture.
    labels – Optional output 2D array of labels (the discrete Voronoi diagram). It has the type
            CV_32SC1 and the same size as src . See the details below.
    labelType – Type of the label array to build. If labelType==DIST_LABEL_CCOMP then each
            connected component of zeros in src (as well as all the non-zero pixels closest to 
            the connected component) will be assigned the same label. If labelType==DIST_LABEL_PIXEL
           then each zero pixel (and all the non-zero pixels closest to it) gets its own label.
'''

import cv2
import sys
import os
import numpy as np

sfile='4.png'
img=cv2.imread(sfile)
height, width = img.shape[:2]  
# 缩小图像  
size = (int(width*0.5), int(height*0.5))  
img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
dist_gray = cv2.distanceTransform(gray, cv2.DIST_L2, 5)
dist_gray_norm=cv2.normalize(dist_gray, 0, dist_gray.max(), cv2.NORM_MINMAX); 
dist_thresh = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
dist_thresh_norm=cv2.normalize(dist_thresh, 0, dist_thresh.max(), cv2.NORM_MINMAX); 

cv2.imshow('img',img)
cv2.imshow('gray',gray)
cv2.imshow('thresh',thresh)
cv2.imshow('dist_gray',dist_gray)
cv2.imshow('dist_gray_norm',dist_gray_norm)
cv2.imshow('dist_thresh',dist_thresh)
cv2.imshow('dist_thresh_norm',dist_thresh_norm)

print(dist_gray.max())
print(dist_gray.min())
print(dist_gray_norm.max())
print(dist_thresh.max())
print(dist_thresh)
np.savetxt('text.txt',dist_thresh)

cv2.waitKey()

