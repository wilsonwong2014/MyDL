#!/usr/bin/env python3
# -*- codeing:utf-8 -*-
'''制作训练样本
图像文件ID集：
    如 训练文件ID集train.txt，每行一个图像ID <img_id>，范例：
        2008_000008
        2008_000015
        2008_000019    
        
原始图像存放目录:JPEGImages
    存放训练/测试所有原始图像<img_id>.jpg，范例：
    2008_000008.jpg

标注文件目录:Annotations
    存放图像文件的标注信息，每个图像文件对应一个标注文件，以<img_id>关联，范例：
    2008_000008.xml
    
检测类别：
    数据集的目标检测类别
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow"
                , "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa"
                , "train", "tvmonitor"]
                
制作训练样本：
    1.读取图像文件集的图像ID
    2.获取图像路径
    3.读取标注信息
    4.制作训练样本
        Row format: image_file_path box1 box2 ... boxN;
        Box format: x_min,y_min,x_max,y_max,class_id (no space).
        范本：
            path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
            path/to/img2.jpg 120,300,250,600,2
    
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
from os import getcwd
import os

#数据集
#sets=[('2012', 'train'), ('2012', 'val'), ('2012', 'test')]
sets=[('2012', 'train'), ('2012', 'val')]

#检测类别
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#数据集根目录
#path='%s/data'%(os.getenv('HOME'))
path='%s/e/dataset_tiptical'%os.getenv('HOME')


def convert_annotation(year, image_id, list_file):
    in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml'%(path,year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

#wd = getcwd()

for year, image_set in sets:
    #/home/hjw/e/dataset_tiptical/VOCdevkit/VOC2012/ImageSets/Main/train.txt
    image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(path,year, image_set)).read().strip().split()
    #/home/hjw/e/dataset_tiptical/VOCdevkit/VOC2012/model_data/2012_train.txt
    list_file = open('%s/VOCdevkit/VOC%s/model_data/%s.txt'%(path,year, image_set), 'w')
    for image_id in image_ids:
        # .../xxxx.jpg xmin,ymin,xmax,ymax,cls_id
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(path, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

