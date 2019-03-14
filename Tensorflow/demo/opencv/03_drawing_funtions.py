#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Drawing Functions in OpenCV
范例：
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
文档：
https://docs.opencv.org/3.4.0/d6/d6e/group__imgproc__draw.html

Goal

Learn to draw different geometric shapes with OpenCV
You will learn these functions : cv2.line(), cv2.circle() , cv2.rectangle(), cv2.ellipse(), cv2.putText() etc.

图形绘制
'''

import numpy as np
import cv2

#Drawing Line
#    img=cv.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]	)
# Create a black image
width=512;
height=512;
depth=3;
pt1=(0,0);
pt2=(511,511);
color=(255,0,0);
thickness=5;
img = np.zeros((width,height,depth), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,pt1,pt2,color,thickness);

#Drawing Rectangle
#    img=cv.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
pt1=(384,0);
pt2=(510,128);
color=(0,255,0);
thickness=3;
cv2.rectangle(img,pt1,pt2,color,thickness);

#Drawing Circle
#    img=cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]]	)
center=(447,63);
radius=63;
color=(0,0,225);
thickness=-1;
cv2.circle(img,(447,63), 63, (0,0,255), -1)

#Drawing Ellipse
#    img=cv.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]	)
#    img=cv.ellipse(img, box, color[, thickness[, lineType]])
center=(256,256);
axes=(100,50);
angle=0;
startAngle=0;
endAngle=180;
color=(255,0,0);
thickness=-1;
cv2.ellipse(img,center,axes,angle,startAngle,endAngle,color,thickness);

#Drawing Polygon
#    img=cv.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]])
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32);
pts = pts.reshape((-1,1,2));
is_closed=True;
color=(0,255,255);
cv2.polylines(img,[pts],is_closed,color);

#Adding Text to Images:
#    img=cv.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]	)
text='OpenCV';
org=(10,500);
font = cv2.FONT_HERSHEY_SIMPLEX;
fontFace=font;
fontScale=4;
color=(255,255,255);
thickness=2;
lineType=cv2.LINE_AA;
cv2.putText(img,text,org, fontFace, fontScale,color,thickness,lineType);

cv2.imshow('Draw Functions',img);
cv2.waitKey(0);
cv2.destroyAllWindows();

