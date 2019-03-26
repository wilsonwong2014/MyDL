#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
 
import pdb
pdb.set_trace()

import xml.etree.ElementTree as ET
import os
#import cPickle
import _pickle as cPickle
import numpy as np

 
def parse_rec(filename): #读取标注的xml文件
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
 
    return objects


 
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    计算AP值，若use_07_metric=true,则用11个点采样的方法，将rec从0-1分成11个点，这些点prec值求平均近似表示AP
    若use_07_metric=false,则采用更为精确的逐点积分方法
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
 
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
 
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
 
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
 

######主函数，计算当前类别的recall和precision    
def voc_eval(det_lines,          #检测结果
             annopath,           #标记目录
             imagesetfile,       #检测图像文件名
             classname,          #筛选类别名称
             cachedir,           #缓存目录
             ovthresh=0.5,       #阈值
             use_07_metric=False #AP计算方式
            ):
    '''
    @param det_splitlines   [list ]某类检测结果文件
        数据： [
                [imagename1, confidence, xmin, ymin, xmax, ymax],  #(图像1的第一个结果)
                [imagename1, confidence, xmin, ymin, xmax, ymax],  #(图像1的第二个结果)
                [imagename2, confidence, xmin, ymin, xmax, ymax],  #(图像2的第一个结果)
              ]
    @param annopath     [str ]标注目录
        annopath.format(imagename) should be the xml annotations file. #xml 标注文件。
        annopath=>'annotations/{}.xml'=>annopath.format('2008_000001')=>'annotations/2008_000001.xml'
    @param imagesetfile [str ]检测图像集文件
        文本文件，每行一个图像文件名,不含扩展名
        该文件格式：
            2008_000001
            2008_000002
    @param classname     [str  ]检测类别名称，用于筛选imagesetfile
    @param cachedir      [str  ]缓存目录，用于存放原始数据集的加载文件
    @param ovthresh      [float]IoU阈值
    @param use_07_metric [bool ]AP计算模式
        Whether to use VOC07's 11 point AP computation 
                (default False) #是否使用VOC07的AP计算方法，voc07是11个点采样。    
    @return rec, prec, ap
        rec ---召回率，向量
        prec---准确率，向量
        ap-----平均准确率,标量
        计算方法：
            检测结果数为:N=5
            按置信度由高到低排序
            TP/FP计算：
                筛选某类的检测结果及该类的gt_bbox
                TP[:],FP[:]初始化为False
                遍历检测结果
                    如果检测bbox与该类gt_bbox的IoU大于阈值,
                    则
                        TP[i]=1
                        虚警处理(同一个gt_bbox在不同的检测结果中出现)：FP[i]=1
                    否则
                        FP[i]=1
                    
            TP:[1, 0, 1, 1, 0],积分值=>TP_int=[1,1,2,3,3]
            FP:[0, 1, 0, 0, 1],积分值=>FP_int=[0,1,1,1,2]
            prec:TP_int/(TP_int+FP_int)=>[1, 1/2, 2/3, 3/4, 3/5]
            rec :TP_int/N=>[1/5, 1/5, 2/5, 3/5, 3/5]
            ap:
                if use_07_metric:
                    # 11 point metric
                    ap = 0.
                    for t in np.arange(0., 1.1, 0.1):
                        if np.sum(rec >= t) == 0:
                            p = 0
                        else:
                            p = np.max(prec[rec >= t])
                        ap = ap + p / 11.
                else:
                    # correct AP calculation
                    # first append sentinel values at the end
                    mrec = np.concatenate(([0.], rec, [1.]))
                    mpre = np.concatenate(([0.], prec, [0.]))

                    # compute the precision envelope
                    for i in range(mpre.size - 1, 0, -1):
                        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                    # to calculate area under PR curve, look for points
                    # where X axis (recall) changes value
                    i = np.where(mrec[1:] != mrec[:-1])[0]

                    # and sum (\Delta recall) * prec
                    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])            
    '''
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
 
    #原始数据集缓存文件 =>[str ] cachefile
    # first load gt 加载ground truth。
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl') #只读文件名称。
    
    #读取所有测试图片名称 =>[list] imagenames
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines() #读取所有待检测图片名。
    imagenames = [x.strip() for x in lines] #待检测图像文件名字存于数组imagenames,长度1000。
 
    #加载原始数据文件 =>[dict] recs{文件名:标注结构体数据}
    if not os.path.isfile(cachefile): #如果只读文件不存在，则只好从原始数据集中重新加载数据
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename)) #parse_rec函数读取当前图像标注文件，返回当前图像标注，存于recs字典（key是图像名，values是gt）
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames))) #进度条。
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        print(type(recs))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f) #recs字典c保存到只读文件。
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f) #如果已经有了只读文件，加载到recs。
 
    #提取类别为classname的原始数据集
    # extract gt objects for this class #按类别获取标注文件，recall和precision都是针对不同类别而言的，AP也是对各个类别分别算的。
    class_recs = {} #当前类别的标注
    npos = 0 #npos标记的目标数量
    for imagename in imagenames:
        #筛选类别为classname的原始数据集 => R    
        R = [obj for obj in recs[imagename] if obj['name'] == classname] #过滤，只保留recs中指定类别的项，存为R。
        #提取bbox,gt
        bbox = np.array([x['bbox'] for x in R]) #抽取bbox
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool) #如果数据集没有difficult,所有项都是0.
 
        #检测结果,默认为False
        det = [False] * len(R) #len(R)就是当前类别的gt目标个数，det表示是否检测到，初始化为false。
        #gt目标数量(排除difficult为True的目标)
        npos = npos + sum(~difficult) #自增，非difficult样本数量，如果数据集没有difficult，npos数量就是gt数量。
        #当前类别标注(不含difficult为True的目标)
        class_recs[imagename] = {'bbox': bbox,           #检测边框
                                 'difficult': difficult, #difficult属性
                                 'det': det              #检测结果
                                }#三个属性值长度相同
        
 
    # read dets 读取检测结果
    #detfile = detpath.format(classname)
    #with open(detfile, 'r') as f:
    #    lines = f.readlines() 
    #splitlines = [x.strip().split(' ') for x in lines] #假设检测结果有20000个，则splitlines长度20000
    splitlines=det_lines
    image_ids = [x[0] for x in splitlines] #检测结果中的图像名，image_ids长度20000，但实际图像只有1000张，因为一张图像上可以有多个目标检测结果
    confidence = np.array([float(x[1]) for x in splitlines]) #检测结果置信度
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) #变为浮点型的bbox。

    # sort by confidence 将20000各检测结果按置信度排序
    sorted_ind = np.argsort(-confidence)           #对confidence的index根据值大小进行降序排列。
    sorted_scores = np.sort(-confidence)           #降序排列。
    print('BB.shape:',BB.shape)
    print('sorted_ind.shape:',sorted_ind.shape)    
    BB = BB[sorted_ind, :]                         #重排bbox，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind] #对image_ids相应地进行重排。
 
    # go down dets and mark TPs and FPs 
    nd = len(image_ids) #注意这里是20000，不是1000
    tp = np.zeros(nd)   # true positive，长度20000
    fp = np.zeros(nd)   # false positive，长度20000
    for d in range(nd): #遍历所有检测结果，因为已经排序，所以这里是从置信度最高到最低遍历
        R = class_recs[image_ids[d]]   #当前检测结果所在图像的所有同类别gt
        bb = BB[d, :].astype(float)    #当前检测结果bbox坐标,1个bbox
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float) #当前检测结果所在图像的所有同类别gt的bbox坐标,含有N个
 
        if BBGT.size > 0: 
            # compute overlaps 计算当前检测结果，与该检测结果所在图像的标注重合率，一对多用到python的broadcast机制
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
 
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
 
            overlaps = inters / uni
            ovmax = np.max(overlaps)  #最大重合率
            jmax = np.argmax(overlaps)#最大重合率对应的gt
 
        if ovmax > ovthresh:#如果当前检测结果与真实标注最大重合率满足阈值
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1. #正检数目+1
                    R['det'][jmax] = 1 #该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
                else: #相反，认为检测到一个虚警
                    fp[d] = 1.
        else: #不满足阈值，肯定是虚警
            fp[d] = 1.
 
    # compute precision recall
    fp = np.cumsum(fp) #积分图，在当前节点前的虚警数量，fp长度
    tp = np.cumsum(tp) #积分图，在当前节点前的正检数量
    rec = tp / float(npos) #召回率，长度20000，从0到1
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth 准确率，长度20000，长度20000，从1到0
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
 
    return rec, prec, ap


import os
import pandas as pd
import numpy as np

#参数
det_pdfile   ='%s/data/VOCdevkit/VOC2012/results/result.txt'%(os.getenv('HOME'))          #检测结果
annopath     ='%s/data/VOCdevkit/VOC2012/Annotations/{}.xml'%(os.getenv('HOME'))          #标注目录
imagesetfile ='%s/data/VOCdevkit/VOC2012/ImageSets/Main/val.txt'%(os.getenv('HOME'))      #图片测试集
classpath    ='%s/data/VOCdevkit/VOC2012/model_data/coco_classes.txt'%(os.getenv('HOME')) #筛选类别名称
cachedir     ='%s/data/VOCdevkit/VOC2012/results/cachedir'%(os.getenv('HOME'))            #缓存目录
ovthresh     =0.5                                                                         #阈值
use_07_metric=False                                                                       #AP计算方式

#目录检测
os.makedirs(cachedir) if not os.path.exists(cachedir) else ''

#读取检测类别集
with open(classpath,'r') as f:
    classesname=[x.strip() for x in f.readlines()]

#加载检测结果
df_det=pd.read_csv(det_pdfile,
                   dtype={'filename':np.str,'score':np.float32,'x':np.float32,'y':np.float32,'w':np.float32,'h':np.float32,'classid':np.float32,'classname':np.str})

#类别筛选
for sname in classesname:
    df_cls=df_det[df_det['classname']==sname]
    det_splitlines=[]
    #print(df.loc[dates[0],'A'])
    nums=df_cls.shape[0]
    print('%s has {%d} bbox!'%(sname,nums))
    if nums==0:
        continue
        
    index=df_cls.index
    for n in range(nums):
        filename=df_cls.loc[index[n],'filename']
        score=df_cls.loc[index[n],'score']
        x=df_cls.loc[index[n],'x']
        y=df_cls.loc[index[n],'y']
        w=df_cls.loc[index[n],'w']
        h=df_cls.loc[index[n],'h']
        clssid=df_cls.loc[index[n],'classid']
        classname=df_cls.loc[index[n],'classname']
        det_splitlines.append([filename,score,x-w/2,y-h/2,x+w/2,y+h/2])
    
    rec, prec, ap=voc_eval(det_splitlines,   #检测结果
             annopath,         #标记目录
             imagesetfile,     #检测图像文件名
             classname,        #筛选类别名称
             cachedir,         #缓存目录
             ovthresh,         #阈值
             use_07_metric     #AP计算方式
            )

