#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

Goal

Learn to read video, display video and save video.
Learn to capture from Camera and display it.
You will learn these functions : cv2.VideoCapture(), cv2.VideoWriter()

视频播放
'''


import numpy as np
import cv2

cap = cv2.VideoCapture('02.avi')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
