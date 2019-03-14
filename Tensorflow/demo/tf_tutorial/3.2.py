#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#2. 三层简单神经网络的前向传播算法.ipynb
#######################################
# ####  1. 三层简单神经网络
import tensorflow as tf

# 1.1 定义变量(权重)
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1)); #权重 
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1)); #权重
x = tf.constant([[0.7, 0.9]]) ;                              #输入向量

# 1.2 定义前向传播的神经网络
a = tf.matmul(x, w1);
y = tf.matmul(a, w2);

# 1.3 调用会话输出结果
sess = tf.Session();
sess.run(w1.initializer)  ; #变量初始化
sess.run(w2.initializer)  ; #变量初始化
print(sess.run(y))  ;       #张量计算
sess.close();

# #### 2. 使用placeholder
x = tf.placeholder(tf.float32, shape=(1, 2), name="input");
a = tf.matmul(x, w1);
y = tf.matmul(a, w2);
sess = tf.Session();
init_op = tf.global_variables_initializer()  ; #初始化所有变量
sess.run(init_op);
print(sess.run(y, feed_dict={x: [[0.7,0.9]]}));#前向计算

# #### 3. 增加多个输入
x = tf.placeholder(tf.float32, shape=(3, 2), name="input");
a = tf.matmul(x, w1);
y = tf.matmul(a, w2);
sess = tf.Session();
#使用tf.global_variables_initializer()来初始化所有的变量
init_op = tf.global_variables_initializer()  ;
sess.run(init_op);
print(sess.run(y, feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]})) ;#小批量输入前向计算

