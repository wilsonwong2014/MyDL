#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt

#在图像上绘制图形
def draw_boxs(img,boxs,thickness=1,texts=None,colors=None):
    '''
    img.shape => (height,width,channels)
    boxs      =>  [[x1,y1,x2,y2],......]
    thickness =>  int
    texts     =>  ['text1','text2',....]
    colors    =>  [(255,0,0),..........]
    '''
    for i,box in enumerate(boxs):
        pt1=(box[0],box[1])
        pt2=(box[2],box[3])
        text='' if texts is None else texts[i]
        color=(255,0,0) if colors is None else colors[i]
        cv2.rectangle(img=img,pt1=pt1,pt2=pt2,color=color,thickness=thickness)
        cv2.putText(img,text=text,org=pt1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=color,thickness=thickness)
    return img


if __name__=='__main__':
    img=np.zeros((600,800,3),dtype=np.uint8)
    boxs=[[50,100,300,200],[100,50,200,300]] #(x1,y1,x2,y2)
    img=draw_boxs(img,boxs,thickness=2,texts=['a','b'],colors=[(255,0,0),(0,255,0)])
    boxs=[[60,130,330,230],[120,60,280,390]] #(x1,y1,x2,y2)
    img=draw_boxs(img,boxs,thickness=2,texts=['a','b'],colors=[(255,0,0),(0,255,0)])
    #plt.figure(figsize=(800/72.,600/72.))
    plt.figure
    plt.imshow(img)
    plt.show()
    print ('OK')
