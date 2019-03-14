#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''绘制椭圆
cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift ]]])
cv2.ellipse(img, box, color[, thickness[, lineType ]])! None
Parameters
    img – Image.
    center – Center of the ellipse.
    axes – Half of the size of the ellipse main axes.
    angle – Ellipse rotation angle in degrees.
    startAngle – Starting angle of the elliptic arc in degrees.
    endAngle – Ending angle of the elliptic arc in degrees.
    box – Alternative ellipse representation via RotatedRect or CvBox2D. 
          This means that the function draws an ellipse inscribed in the rotated rectangle.
    color – Ellipse color.
    thickness – Thickness of the ellipse arc outline, if positive. 
          Otherwise, this indicates that a filled ellipse sector is to be drawn.
    lineType – Type of the ellipse boundary. See the line() description.
    shift – Number of fractional bits in the coordinates of the center and values of axes.
'''

import cv2
import numpy as np
W=512
center=(W//2,W//2)
axes=(200,100)
angle=30
thickness = 2
line_type = 8


width=W
height=W
depth=3
img=np.zeros((width,height,depth),np.uint8)

cv2.ellipse(img,         #待绘制的图像
    center,              #椭圆中心
    axes,                #椭圆长轴和短轴的长度
    angle,               #长轴倾角   
    0,                   #弧度绘制的起始角度 
    360,                 #弧度绘制的截止角度
    (255, 0, 0),         #线条的颜色(BGR)
    thickness,           #线宽
    line_type)           #线型

cv2.imshow('img',img)
cv2.waitKey()


