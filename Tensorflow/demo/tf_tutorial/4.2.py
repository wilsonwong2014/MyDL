#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#2. 学习率的设置.ipynb
########################

# #### 假设我们要最小化函数  $y=x^2$, 选择初始点   $x_0=5$
# #### 1. 学习率为1的时候，x在5和-5之间震荡。
import tensorflow as tf
TRAINING_STEPS = 10 ; #迭代次数
LEARNING_RATE = 1   ; #学习率
#输入标量
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x");
#输出标量
y = tf.square(x);
#梯度下降法优化方法
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y);
#-------------------------------
#训练模型
with tf.Session() as sess:
    #初始化所有变量
    sess.run(tf.global_variables_initializer());
    for i in range(TRAINING_STEPS):
        #执行优化
        sess.run(train_op)
        #计算优化参数
        x_value = sess.run(x)
        print("After %s iteration(s): x%s is %f."% (i+1, i+1, x_value) );

# #### 2. 学习率为0.001的时候，下降速度过慢，在901轮时才收敛到0.823355。
TRAINING_STEPS = 1000 ; #迭代次数
LEARNING_RATE = 0.001 ; #学习率
#输入标量
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x");
#输出标量
y = tf.square(x)
#梯度下降优化方法
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y);
#-------------------------------
#训练模型
with tf.Session() as sess:
    #初始化所有变量
    sess.run(tf.global_variables_initializer());
    for i in range(TRAINING_STEPS):
        #执行优化
        sess.run(train_op);
        if i % 100 == 0: 
            #优化参数
            x_value = sess.run(x);
            print("After %s iteration(s): x%s is %f."% (i+1, i+1, x_value));

# #### 3. 使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得不错的收敛程度。
TRAINING_STEPS = 100 ;#迭代次数
global_step = tf.Variable(0);
#指数衰减:c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True);
#　　learning_rate:初始学习率
#  global_      :全局迭代次数
#  decay_steps  :衰减速度，每迭代decay_steps执行一次指数衰减
#  decay_rate   :衰减系数
#  staircase(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True);
#输入标量
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x");
#输出标量
y = tf.square(x);
#梯度下降训练方法
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step);
#-------------------------------
#训练模型
with tf.Session() as sess:
    #初始化所有变量
    sess.run(tf.global_variables_initializer());
    for i in range(TRAINING_STEPS):
        #执行优化
        sess.run(train_op);
        if i % 10 == 0:
            LEARNING_RATE_value = sess.run(LEARNING_RATE);
            global_step_value=sess.run(global_step);
            #优化参数
            x_value = sess.run(x);
            print("After %s iteration(s): x%s is %f, learning rate is %f,global_step:%s" % (i+1, i+1, x_value, LEARNING_RATE_value,global_step_value));


