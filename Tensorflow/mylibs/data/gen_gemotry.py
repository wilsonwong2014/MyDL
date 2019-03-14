#!/usr/bin/env python3
'''随机构造几何图形
    1.构造随机直线
    2.构造随机多边形
    3.构造随机圆形
    4.构造随机椭圆
'''

import os
import numpy as np
import cv2

#构造直线
def gen_line(imgsize=(224,224,3)):
    w=imgsize[1]
    h=imgsize[0]
    img=np.zeros(imgsize,dtype=np.uint8)
    pt1=(np.random.randint(0,w//2-w//4),np.random.randint(0,h))
    pt2=(np.random.randint(w//2+w//4),np.random.randint(0,h))
    thickness=np.random.randint(1,10)
    #cv2.line(img=img,pt1=pt1,pt2=pt2,color=(255,0,0),thickness=thickness)
    cv2.line(img=img,pt1=pt1,pt2=pt2,color=(255,0,0),thickness=thickness)
    return img


#构造多边形
def gen_poly(imgsize=(224,224,3),N=3,fill=False):
    w=imgsize[1]
    h=imgsize[0]
    #构建初始图形
    img=np.zeros(imgsize,dtype=np.uint8) 
    #随机设置线条大小
    thickness=np.random.randint(1,10) 
    #随机半径选择
    R=w//2 if w<h else h//2
    R=np.random.randint(30,R-10)
    #在半径为R的圆上随机选择N个点
    theta_step=2*np.pi/N
    thetas=[i*theta_step for i in range(N)]
    thetas=[theta + np.random.randint(-30,30)*theta_step/100.0 for theta in thetas]
    pts=[(R*np.cos(theta),R*np.sin(theta)) for theta in thetas]
    pts=np.array(pts).astype(np.int32)
    pts[:,0]+=w//2
    pts[:,1]+=h//2
    pts=pts.reshape((-1,1,2))
    '''这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
       如果第三个参数是 False，我们得到的多边形是不闭合的（首尾不相连）。
    ''' 
    if fill:
        #填充
        cv2.fillPoly(img,[pts],color=(255,0,0))
    cv2.polylines(img,[pts],True,(255,0,0),thickness=thickness) # 图像，点集，是否闭合，颜色，线条粗细
    return img


#构造圆
def gen_circle(imgsize=(224,224,3),fill=False):
    w=imgsize[1]
    h=imgsize[0]
    thickness=np.random.randint(1,10)
    R=np.random.randint(30,w//2-10) if w<h else np.random.randint(30,h//2-10)
    img=np.zeros(imgsize,dtype=np.uint8)
    if fill:
        thickness=-1
    cv2.circle(img=img,center=(w//2,h//2),radius=R,color=(255,0,0),thickness=thickness)
    return img


#构造椭圆
def gen_ellipse(imgsize=(224,224,3),fill=False):
    w=imgsize[1]
    h=imgsize[0]
    thickness=np.random.randint(1,10)
    if fill:
        thickness=-1
    a=np.random.randint(30,w//2-10)
    b=np.random.randint(30,h//2-10)
    angle=np.random.randint(0,360)
    img=np.zeros(imgsize,dtype=np.uint8)
    cv2.ellipse(img=img,center=(w//2,h//2),axes=(a,b),angle=angle,startAngle=0,endAngle=360,color=(255,0,0),thickness=thickness)
    return img


if __name__ == '__main__':
    img=gen_line()
    cv2.imshow('gen_line',img)

    img=gen_poly()
    cv2.imshow('gen_poly',img)

    img=gen_circle()
    cv2.imshow('gen_circle',img)

    img=gen_ellipse()
    cv2.imshow('gen_ellipse',img)

    cv2.waitKey()            #等待退出-必须
    cv2.destroyAllWindows()  #销毁窗口资源  

