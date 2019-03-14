#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

##########################
# 张量,常量,变量 使用范例

#######################
# 常量使用范例
#constant(
#    value,
#    dtype=None,         #参考tf.DType
#    shape=None,
#    name='Const',
#    verify_shape=False  #维数(shape)是否可以修改
#)
#######################
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

#############################
# 变量使用范例
#__init__(
#    initial_value=None,
#    trainable=True,
#    collections=None,
#    validate_shape=True,
#    caching_device=None,
#    name=None,
#    variable_def=None,
#    dtype=None,
#    expected_shape=None,
#    import_scope=None
#)
with tf.Session() as sess:
    #定义变量
    v1=tf.Variable(initial_value=1.0);
    sess.run(v1.initializer);            #变量初始化
    tf.initialize_all_variables().run(); #初始化所有变量
    print("variable properties:");
    print("  device=>",v1.device);
    print("  dtype=>",v1.dtype);
    print("  graph=>",v1.graph);
    print("  initial_value=>",v1.initial_value);
    print("  initializer=>",v1.initializer);
    print("  name=>",v1.name);
    print("  op=>",v1.op);
    print("  shape=>",v1.shape);
 
    print(sess.run(v1));



