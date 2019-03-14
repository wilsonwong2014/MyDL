#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
Goals

Here, you will learn how to read an image, how to display it and how to save it back
You will learn these functions : cv2.imread(), cv2.imshow() , cv2.imwrite()
Optionally, you will learn how to display images with Matplotlib

plt 与 cv2.imshow不能同时使用
图像文件读写显示
'''

import sys
import cv2
import numpy as np

print(len(sys.argv));
is_use_plt=False;
if(len(sys.argv)>1):
    is_use_plt=sys.argv[1]=='1';

if(is_use_plt):
    import matplotlib.pyplot as plt
    print('Matplotlib');
    #Using Matplotlib
    #Matplotlib is a plotting library for Python which gives you wide variety of plotting methods. You will see them in coming articles. Here, you will learn how to display image with Matplotlib. You can zoom images, save it etc using Matplotlib.
    img = cv2.imread('1.jpg',0);
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic');
    plt.xticks([]), plt.yticks([]);  # to hide tick values on X and Y axis
    plt.show();
else:
    print('cv2');
    #Load an color image in grayscale
    img=cv2.imread('1.jpg',0);

    #Display image
    cv2.imshow('image',img);
    #Write an image
    cv2.imwrite('1_new.jpg',img);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

