#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 绘制直线

Python: cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift ]]])
Python: cv.Line(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)

Parameters
    img – Image.
    pt1 – First point of the line segment.
    pt2 – Second point of the line segment.
    color – Line color.
    thickness – Line thickness.
    lineType – Type of the line:
        – 8 (or omitted) - 8-connected line.
        – 4 - 4-connected line.
        – CV_AA - antialiased line.
    shift – Number of fractional bits in the point coordinates.
'''

import cv2
#import cv
import numpy as np

width=512
height=512
depth=3
pt1=(0,0)
pt2=(200,200)
color=(255,0,0)
thickness=1
lineType=8

#cv2
img1 = np.zeros((width,height,depth), np.uint8)
cv2.line(img1,pt1,pt2,color,thickness,lineType)
cv2.imshow("img1",img1)
cv2.waitKey()
#cv

