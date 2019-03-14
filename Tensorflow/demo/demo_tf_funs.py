#!/usr/bin/env python3
# -*- codeing:utf-8 -*-

################################
#      汇总基础操作单元         #
################################
#
#
#
################################


#引用
import tensorflow as tf
import numpy as np

################################
#          定义常量            #
################################
val1 = tf.constant([[1., 2.,3.],[4.,5.,6.]]);   #2x3数组,返回的是张量Tensor
val2 = tf.constant([[1., 2.],[3.,4.],[5.,6.]]); #3x2数组,返回的是张量Tensor

################################
#          定义乘法            #
################################
mul1_2=tf.matmul(val1,val2);   # mul1_2=val1*val2;返回的是张量Tensor

###############################
#       创建会话执行图         #
###############################
sess=tf.Session();
matVal=sess.run(mul1_2);      #执行图(默认),返回numpy.narray
sess.close();

###############################
#       with 示例             #
###############################
with tf.Session() as sess:
    matVal = sess.run(mul1_2);

##############################
#   变量定义与初始化          #
##############################
with tf.Session() as sess:
    v1=tf.Variable(1,name="v1"); #定义变量
    sess.run(v1.initializer);    #变量初始化
    print(v1);
    print(sess.run(v1));

