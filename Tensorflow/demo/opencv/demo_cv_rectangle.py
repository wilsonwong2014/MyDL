#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 绘制矩形
cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift ]]])
cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)

Parameters
    img – Image.
    pt1 – Vertex of the rectangle.
    pt2 – Vertex of the rectangle opposite to pt1 .
    rec – Alternative specification of the drawn rectangle.
    color – Rectangle color or brightness (grayscale image).
    thickness – Thickness of lines that make up the rectangle. 
       Negative values, like CV_FILLED , mean that the function has to draw a filled rectangle.
    lineType – Type of the line. See the line() description.
    shift – Number of fractional bits in the point coordinates.
'''
import cv2
import numpy as np

width=512
height=512
depth=3
img=np.zeros((width,height,depth),np.uint8)

pt1=(20,20)
pt2=(200,200)
color=(255,0,0)
thickness=1
lineType=8
cv2.rectangle(img,pt1,pt2,color,thickness,lineType)
cv2.imshow('img',img)
cv2.waitKey()

