#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#3. 正则化.ipynb
###########################

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# #### 1. 生成模拟数据集。
data = [] ;
label = [];
np.random.seed(0);#随机数种子,生成随机数np.random.random();
# 以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
for i in range(150):
    x1 = np.random.uniform(-1,1); #生成均匀分布随机数
    x2 = np.random.uniform(0,2);  #生成均匀分布随机数
    if x1**2 + x2**2 <= 1:
        #生成正态分布随机数,nx2
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)]);
        label.append(0);
    else:
        #生成正态分布随机数,nx2
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)]);
        label.append(1);
    
data = np.hstack(data).reshape(-1,2);    #把data排成一行,然后重排为nx2
label = np.hstack(label).reshape(-1, 1); #把data排成一行,然后重排为nx1

#绘制数据
#PYthon——plt.scatter各参数详解
#   https://blog.csdn.net/qiu931110/article/details/68130199

#plt.scatter(data[:,0], data[:,1], c=label,
#           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white");
#报错ValueError: c of shape (150, 1) not acceptable as a color sequence for x with size 150, y with size 150
#修正:由于label.shape=>(n,1);使用np.squeeze从数组的形状中删除单维条目，即把shape中为1的维度去掉
#    np.squeeze(label).shape=>(n,)
plt.scatter(data[:,0], data[:,1], c=np.squeeze(label),
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white");
plt.show();


# #### 2. 定义一个获取权重，并自动加入正则项到损失的函数。
def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32);
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var));
    return var

# #### 3. 定义神经网络。
x  = tf.placeholder(tf.float32, shape=(None, 2));
y_ = tf.placeholder(tf.float32, shape=(None, 1));
sample_size = len(data);

# 每层节点的个数
layer_dimension = [2,10,5,3,1];
n_layers = len(layer_dimension);

cur_layer = x; #当前层
in_dimension = layer_dimension[0];#输入节点数

# 循环生成网络结构
for i in range(1, n_layers):
    out_dimension = layer_dimension[i]; #输出节点数
    weight = get_weight([in_dimension, out_dimension], 0.003);  #定义权重并加入"losses"集合
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]));#定义偏置量
    cur_layer = tf.nn.elu(tf.matmul(cur_layer, weight) + bias); #构造网络
    in_dimension = layer_dimension[i];                          #下一层的输入节点数

#最后一层即位输出层
y= cur_layer;

# 损失函数的定义:最小均方差。
mse_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / sample_size;
tf.add_to_collection('losses', mse_loss);
loss = tf.add_n(tf.get_collection('losses'));


# #### 4. 训练不带正则项的损失函数mse_loss。
# 定义训练的目标函数mse_loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(mse_loss);
TRAINING_STEPS = 40000;
#训练
with tf.Session() as sess:
    #初始化所有变量
    tf.global_variables_initializer().run();
    for i in range(TRAINING_STEPS):
        #优化
        sess.run(train_op, feed_dict={x: data, y_: label});
        if i % 2000 == 0:
            print("After %d steps, mse_loss: %f" % (i,sess.run(mse_loss, feed_dict={x: data, y_: label})));

    # 画出训练后的分割曲线       
    xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01];
    grid = np.c_[xx.ravel(), yy.ravel()];
    probs = sess.run(y, feed_dict={x:grid});
    probs = probs.reshape(xx.shape);

plt.scatter(data[:,0], data[:,1], c=np.squeeze(label),
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white");
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1);
plt.show();


# #### 5. 训练带正则项的损失函数loss。
# 定义训练的目标函数loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(loss);
TRAINING_STEPS = 40000;
#训练
with tf.Session() as sess:
    #初始化所有变量
    tf.global_variables_initializer().run();
    for i in range(TRAINING_STEPS):
        #优化
        sess.run(train_op, feed_dict={x: data, y_: label});
        if i % 2000 == 0:
            print("After %d steps, loss: %f" % (i, sess.run(loss, feed_dict={x: data, y_: label})));

    # 画出训练后的分割曲线       
    xx, yy = np.mgrid[-1:1:.01, 0:2:.01];
    grid = np.c_[xx.ravel(), yy.ravel()];
    probs = sess.run(y, feed_dict={x:grid});
    probs = probs.reshape(xx.shape);

plt.scatter(data[:,0], data[:,1], c=np.squeeze(label),
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white");
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1);
plt.show();

