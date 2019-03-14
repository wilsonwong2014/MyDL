#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Lambda层
tf.keras.layers.Lambda:
__init__(
    function,
    output_shape=None,
    mask=None,
    arguments=None,
    **kwargs
)
'''

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Lambda
from keras import backend as K

#创建序列模型
model=Sequential() 
print('len(model.layers):',len(model.layers)) #=>0,初始状态层数为0

# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2,input_shape=(2,3),name='lambda_1'))

# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

#Lambda层函数体
def antirectifier(x):
    x -= K.mean(x, axis=-1, keepdims=True)
    x = K.l2_normalize(x, axis=-1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=-1)

model.add(Lambda(antirectifier,name='lambda_2'))

#输出模型序列信息
for layer in model.layers:
    print('layer name:',layer.name)
    print('layer.input:',layer.input)
    print('layer.output:',layer.output)

