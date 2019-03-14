#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
keras 30分钟快速上手
  https://keras-cn.readthedocs.io/en/latest/
'''

import os;
import sys;
############# 调试 begin ###############
argc = len(sys.argv);
import pdb       
if argc>1 and sys.argv[1]=='dbg':    
    pdb.set_trace(); #调试
############# 调试 end   ###############

#引入模块
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

##构造样本数据
#   构造二维样本数据data[n x 3],进行二分类
#   每样表示一个样本采集,第一列是特征1,第二列是特征2,第三列为标签
#   特征值范围[0...1]
#   标签[0,1]---特征1+特征2<1时为0,否则为1
feature_num=2;    #特征数
cols=feature_num+1;
examples_num=1000;#样本数
train_percent=0.8;#训练样本比例
data=np.random.random([examples_num,cols]);            #生成随机矩阵
idx=np.where(data[:,0:cols].sum(1)>feature_num);       #特征值之和大于1设为标签1
data[idx,cols-1]=1;
idx=np.where(data[:,0:cols].sum(1)<=feature_num);      #特征值之和小于等于1设为标签0
data[idx,cols-1]=0;
#训练数据集
train_x=data[0:int(examples_num*train_percent),0:cols-1];
train_y=np.to_categorical(data[0:int(examples_num*train_percent),cols-1].astype(int));
#评估数据集
#测试数据集
test_x=data[int(examples_num*train_percent):,0:cols-1];
test_y=np.to_categorical(data[int(examples_num*train_percent):,cols-1].astype(int));

##创建模型
model=Sequential();
model.add(Dense(units=4, input_dim=feature_num));
model.add(Activation("relu"));
model.add(Dense(units=1));
model.add(Activation("softmax"));
#模型编译
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']);
##模型训练
model.fit(train_x, train_y, epochs=5, batch_size=32);
##模型评估
loss_and_metrics = model.evaluate(test_x, test_y, batch_size=128);
##模型预测
classes = model.predict(test_x, batch_size=128);

