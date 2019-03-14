#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#######################
# 常量使用范例
#constant(
#    value,
#    dtype=None,         #参考tf.DType
#    shape=None,
#    name='Const',
#    verify_shape=False  #维数(shape)是否可以修改
#)

import tensorflow as tf
###################
with tf.Session() as sess:
    #定义标量
    t1=tf.constant(1.0,dtype="float32",name="const1");
    print(t1);          #返回的是张量
    print(sess.run(t1));#sess.run(t1)=>numPy数组
    
    #定义向量
    t2=tf.constant([1.0,2.0],dtype="float32",name="const2");
    print(t2);          #返回的是张量
    print(sess.run(t2));#sess.run(t2)=>numPy数组    

    #定义矩阵
    t3=tf.constant([[1.0,2.0],[3.0,4.0]],dtype="float32",name="const3");
    print(t3);          #返回的是张量
    print(sess.run(t3));#sess.run(t3)=>numPy数组        

