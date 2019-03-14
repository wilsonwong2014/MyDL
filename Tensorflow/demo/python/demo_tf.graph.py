#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''TensorFlow计算图操作
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5046 (Dense)           (None, 700)               549500    
_________________________________________________________________
dropout_3917 (Dropout)       (None, 700)               0         
_________________________________________________________________
dense_5047 (Dense)           (None, 512)               358912    
_________________________________________________________________
dropout_3918 (Dropout)       (None, 512)               0         
_________________________________________________________________
dense_5048 (Dense)           (None, 256)               131328    
_________________________________________________________________
dropout_3919 (Dropout)       (None, 256)               0         
_________________________________________________________________
dense_5049 (Dense)           (None, 400)               102800    
_________________________________________________________________
dropout_3920 (Dropout)       (None, 400)               0         
_________________________________________________________________
dense_5050 (Dense)           (None, 10)                4010      
=================================================================
Total params: 1,146,550
Trainable params: 1,146,550
Non-trainable params: 0
'''

import tensorflow as tf
import numpy as np

#构造计算图
v1=tf.Variable(0,name='v1')
v2=tf.Variable(1,name='v2')
v3=v1+v2

x=tf.Variable(np.random.randn(3,4),name='x')
w=tf.Variable(np.random.randn(4,2),name='w')
b=tf.Variable(np.random.randn(2),name='b')

#创建会话
sess=tf.InteractiveSession()

#初始化所有变量
tf.global_variables_initializer().run()

#遍历所有tf.Operation
print('tf.get_defualt_graph().get_operations():')
for op in tf.get_default_graph().get_operations():
    #print('name:%s,type:%s'%(op.name,op.type))
    #{:<10d} 
    print('name:{:<20},type:{:<20}'.format(op.name,op.type))
    #print('inputs:',op.inputs)
    #print('outputs:',op.outputs)


#print('tf.get_default_graph().as_graph_def().node:')
#for n in tf.get_default_graph().as_graph_def().node:
#    print('name:',n.name)

