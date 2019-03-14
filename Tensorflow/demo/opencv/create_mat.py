#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
创建mat保存
'''

import cv2
import numpy as np
rows=100
cols=100
image = np.zeros((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        image[i,j]=i+j

cv2.imwrite('1.png',image)
