#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''制作anchors文件
    通过训练样本数据文件train.txt制作anchors文件，训练样本文件由 voc_annotations.py制作。

样本数据文件格式    ：<img_file> <space> <Box1> <space> <Box2> ... <BoxN>
Box格式(不能有空格): <xmin>,<ymin>,<xmax>,<ymax>,<cls_id>
      范本：
      /path/1.jpg 10,30,100,200,0 110,130,200,400,1
      
anchors文件格式: <width>,<height>[,<width>,<height> .....]
    范本：
         12,35,300,200,.......
'''
import os
import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        #self.filename = "2012_train.txt"
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        '''kmeans聚类算法
            np.median:计算沿指定轴的中位数
        1.随机选取k个聚类中心点
        2.计算外框分别到k个中心点距离
        3.选取最近的中心点=>current_nearest
        4.判断current_nearest与last_nearest是否相同，相同表示聚类完毕，退出!
        5.更新聚类中心点
            5.1. 选取聚类为curster的所有外框数据.
            5.2. 计算中位数[w,h].
            5.3. 更新聚类中心点 clusters[cluster]=[w,h]
        6.保存分类结果 last_nearst=current_nearest，跳到[2]继续执行.
        
        返回值：[k x 2]
            k --- k个聚类外框
            第一列为外框宽度
            第二列为外框高度
        '''
        box_number = boxes.shape[0]            #提取外框个数
        distances = np.empty((box_number, k))  #
        last_nearest = np.zeros((box_number,)) #
        np.random.seed()
        #随机抽取k个中心点=>(k,)
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            #距离计算=>(box_number,k),行表示外框，列表示分类中心点，即：第i个外框到第j个分类的距离.
            distances = 1 - self.iou(boxes, clusters)       

            current_nearest = np.argmin(distances, axis=1)  #提取最近点=>(box_number,),即：第i个外框离第j个分类最近.
            #分类完毕，退出循环
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            #更新聚类中心点
            for cluster in range(k):
                #计算第cluster个分类中心点，并更新.
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)
                
            #保存最后分类结果
            last_nearest = current_nearest
        #返回聚类结果
        return clusters

    def result2txt(self, data):
        '''保存外框聚类结果到文件
        文件格式：Box1[,Box2[,....]
                Box:width,height
        范例：
            13,14,35,400,......
        '''
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        '''读取训练样本，提取所有标注框信息
        返回：[n x 2]
            n --- 标注框个数
            第一列为标注框宽度
            第二列为标注框高度
        '''
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ") #第0个为文件路径，之后元素为各个标注框信息(xmin,ymin,xmax,ymax,cls_id)
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()                           #提取所有标注框信息
        result = self.kmeans(all_boxes, k=self.cluster_number) #kmean聚类分析
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "%s/work/data/yolo/mytrain_model_data/train.txt"%os.getenv('HOME')  #由voc_annotations.py制作
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
