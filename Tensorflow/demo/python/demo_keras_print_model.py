#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''输出模型图像png
https://keras-cn.readthedocs.io/en/latest/
Keras:基于python的深度学习库
'''

###########################################
import os
import sys
#数据目录
data_path='%s/data/demo/%s'%(os.getenv('HOME'),sys.argv[0].split('.')[0])
os.makedirs(data_path) if not os.path.exists(data_path) else None

'''
快速开始：30s上手Keras
'''
#构造样本数据
import numpy as np
#样本数
samples=1000;
x=np.random.random([samples,100]);
y=np.ndarray(shape=(samples,10),dtype=np.int32);

x_train=x;
y_train=y;
x_test=x;
y_test=y;
#Sequential模型定义
from keras.models import Sequential
model=Sequential();
print("model=Sequential();=>");
print(model);

#将一些网络层通过add()堆叠起来,构成一个模型
from keras.layers import Dense,Activation
model.add(Dense(units=64,input_dim=100));
model.add(Activation("relu"));
model.add(Dense(units=10));
model.add(Activation("softmax"));

#编译模型
model.compile(loss="categorical_crossentropy",optimizer="sgd",metrics=["accuracy"]);

#训练网络
model.fit(x_train,y_train,epochs=5,batch_size=32);

#模型评估
loss_and_metrics=model.evaluate(x_test,y_test,batch_size=128);

#数据预测
classes=model.predict(x_test,batch_size=128);

###########################################
#打印模型
from keras.utils import plot_model
plot_model(model, to_file='%s/model.png'%(data_path));
#显示模型
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
lena = mpimg.imread('%s/model.png'%(data_path)) # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

