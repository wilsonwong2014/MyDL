#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
#     matplotlib范例       #
############################
import cv2
import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float));
x = np.linspace(-10, 10, 30);
y = func(x);
plt.plot(x, y);
plt.xlabel('x');
plt.ylabel('y(x)');
plt.show();

#import matplotlib.pyplot as plt
#plt.figure(1)   #创建图表1
#plt.figure(2)      #创建图表2
#plt.show()          #显示所有图表
