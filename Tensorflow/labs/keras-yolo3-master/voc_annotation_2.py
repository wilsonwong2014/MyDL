#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''制作yolo训练样本
    把xml格式的注释文件转换为yolo训练样本格式的文件.
    xml格式的注释文件，如“附件:XML注释文件范本”.
    yolo训练样本格式文件，如“附件:yolo训练样本文件范本”
    标记分类文件，如“附件:标记分类文件范本”

使用范例：
    范例1：指定输入输出文件格式
    $python3 voc_annotations_2.py --imgs_path ./imgs --annotations_path ./annotations --classes_file ./classes_file.txt --out_file ./out_file.txt --ids_file ./ids.txt
    范例2：子集分割
    $python3 voc_annotations_2.py --imgs_path ./imgs --annotations_path ./annotations --classes_file ./classes_file.txt --out_path ./out_path --split_subset "train:0.6,valid:0.2,test:0.2"


附件：标记分类文件范本
-------------------------
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor


附件：yolo训练样本文件范本
-------------------------
    Generate your own annotation file and class names file.
    One row for one image;
    Row format: image_file_path box1 box2 ... boxN;
    Box format: x_min,y_min,x_max,y_max,class_id (no space).
    For VOC dataset, try python voc_annotation.py
    Here is an example:

    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...


附件：XML注释文件范本
-----------------------
<annotation>
	<folder>VOC2012</folder>
	<filename>2007_000027.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
	</source>
	<size>
		<width>486</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>174</xmin>
			<ymin>101</ymin>
			<xmax>349</xmax>
			<ymax>351</ymax>
		</bndbox>
		<part>
			<name>head</name>
			<bndbox>
				<xmin>169</xmin>
				<ymin>104</ymin>
				<xmax>209</xmax>
				<ymax>146</ymax>
			</bndbox>
		</part>
		<part>
			<name>hand</name>
			<bndbox>
				<xmin>278</xmin>
				<ymin>210</ymin>
				<xmax>297</xmax>
				<ymax>233</ymax>
			</bndbox>
		</part>
		<part>
			<name>foot</name>
			<bndbox>
				<xmin>273</xmin>
				<ymin>333</ymin>
				<xmax>297</xmax>
				<ymax>354</ymax>
			</bndbox>
		</part>
		<part>
			<name>foot</name>
			<bndbox>
				<xmin>319</xmin>
				<ymin>307</ymin>
				<xmax>340</xmax>
				<ymax>326</ymax>
			</bndbox>
		</part>
	</object>
</annotation>

'''
import xml.etree.ElementTree as ET
import os
import sys
import shutil
import pdb       
import numpy as np
import cv2
import argparse
from mylibs import funs

#获取检测注释信息
def get_annotation(xml_file,classes):
    '''获取检测注释信息
    @param xml_file 注释xml文件
    @param classes  分类list
    @return img_file,[(xmin,ymin,xmax,ymax,clsid),......]
    范例:
        
    '''
    in_file = open(xml_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    img_file='%s/%s'%(root.find('folder').text,root.find('filename').text)
    list_box_clsid=[]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text  #标记质量
        cls = obj.find('name').text             #标记分类名称
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)             #标记分类ID
        xmlbox = obj.find('bndbox')             #bounding box
        list_box_clsid.append([int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text),cls_id]])
    return img_file,list_box_clsid
#xml注释文件转换
def convert_annotation_1(out_file,ids_file,imgs_path,annotations_path,classes):
    '''xml注释文件转换
    输出文件格式：
        imgfile,xmin,ymin,xmax,ymax,clsid,......
        ......
    @param out_file  输出文件
    @param ids_file  图像ID文件
    @param imgs_path 图像存放目录
    @param annotations_path 注释文件存放目录    
    @param classes   分类名list
    '''
    image_ids = open(ids_file).read().strip().split()     #分离图像ID
    list_file = open(out_file, 'w')                       #转换输出文件
    for image_id in image_ids:
        xml_file='%s/%s.xml'%(annotations_path,image_id)  #XML文件
        img_file='%s/%s.jpg'%(imgs_path,image_id)         #图像路径
        img_annotation=get_annotation(xml_file,classes)   #图像标记
        list_file.write(img_file)                         #写文件
        for box in img_annotation[1]:
            list_file.write(" " + ",".join([str(a) for a in box]))
        list_file.write('\n')
    list_file.close()

#xml注释文件转换
def convert_annotation_2(out_path,imgs_path,annotations_path,classes,split_subset):
    '''xml注释文件转换
    输出文件格式：
        imgfile,xmin,ymin,xmax,ymax,clsid,......
        ......
    @param out_path  输出目录
    @param imgs_path 图像存放目录
    @param annotations_path 注释文件存放目录    
    @param classes   标记分类名list,[cat,dog,....]
    @param split_subset  分类子集，如:{"train":0.6,"valid":0.2,"test":0.2}或{"train":60,"valid":20,"test":20}
    '''
    #图像分类
    split_class=split_class.split(',')
    #图像分类比例
    split_per=[float(x) for x in split_per.split(',')]
    #检索文件
    files=os.listdir(annotations_path)
    #注释文件总数
    num_files=len(files)
    prev_index=0
    for k,v in split_subset.items():
        out_file='%s/%s.txt'%(out_path,k)           #输出文件
        #子集
        sub_files=files[prev_index:int(v) if v>1 else int(v*num_files)]  
        list_file = open(out_file, 'w')                       #转换输出文件
        for xml_file in sub_files:
            img_file,img_annotation=get_annotation(xml_file,classes)   #图像标记
            list_file.write('%s/%s'%(imgs_path,img_file))              #写文件
            for box in img_annotation:
                list_file.write(" " + ",".join([str(a) for a in box]))
            list_file.write('\n')
        list_file.close()


def params():
    ''' 程序参数
    '''
    #程序描述
    description='xml注释文件转换'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument[公共参数部分]
    parser.add_argument('--imgs_path',type=str, help='图像根目录.   eg. --imgs_path "./ImagesSet"',default='');
    parser.add_argument('--annotations_path',type=str, help='XML注释文件目录.   eg. --annotations_path "./annotations"',default='');
    parser.add_argument('--classes_file',type=str, help='标记类别文件.   eg. --src "./classes_file.txt"',default='');
    # Add argument[指定输入输出文件]
    parser.add_argument('--out_file',type=str, help='输出文件.   eg. --out_file "./out.txt"',default='');
    parser.add_argument('--ids_file',type=str, help='图像ID文件. eg. --ids_file "./ids.txt"',default='');
    # Add argument[训练集分割]
    parser.add_argument('--out_path',type=str, help='输出目录.   eg. --out_path "./out_path"',default='');
    parser.add_argument('--split_subset',type=str, help='子集分类. eg. --split_subset "train:0.6,valid:0.2,test:0.2"',default='');
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    flag=1 if arg.out_file=='' else 0  #功能选择
    #基本参数
    imgs_path=arg.imgs_path
    annotations_path=arg.annotations_path
    classes_file=arg.classes_file
    if classes_file=='':
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    else:    
        classes = open(classes_file).read().strip().split()
    
    if flag==1:
        #子集分割
        out_path=arg.out_path
        split_subset={}
        for k,v in dict([s.split(':') for s in arg.split_subset.split(',')]).items():
            split_subset[k]=float(v)
        convert_annotation_2(out_path,imgs_path,annotations_path,classes,split_subset)
    else:
        #指定输入输出文件
        out_file=arg.out_file
        ids_file=arg.ids_file
        convert_annotation_1(out_file,ids_file,imgs_path,annotations_path,classes)
    

if __name__=='__main__':
    arg=params()
    main(arg)

