#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''输出文本
'''

import numpy as np
import cv2

img1=np.zeros((512,512,3),dtype=np.uint8) #随机生成0-1的浮点图像
print(img1.shape)
print(img1.dtype)
#img2=img1*100                  #图像值增大100倍,大于1的显示为白色(255)

#cv2.imshow('rand',img2)
#cv2.putText( img, text, org, fontFace, fontScale, color [ , thickness [ , lineType [ , bottomLeftOrigin ]]] )
cv2.putText(img1,'1',(100,100),'FreeMono',1,(255,255,255),2)
#cv2.putText(img1,'好',(250,250),cv2.FONT_HERSHEY_COMPLEX,6,(255,255,255),25)
cv2.imshow('src',img1)
cv2.waitKey()
