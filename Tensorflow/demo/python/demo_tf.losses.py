#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

'''损失函数 tf.losses
'''

import os
import sys
import tensorflow as tf
import numpy as np
import keras

'''
tf.losses.sigmoid_cross_entropy(
    multi_class_labels,
    logits,
    weights=1.0,
    label_smoothing=0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
Args:
    multi_class_labels: [batch_size, num_classes] target integer labels in {0, 1}.
    logits: Float [batch_size, num_classes] logits outputs of the network.
    weights: Optional Tensor whose rank is either 0, or the same rank as labels, and must be broadcastable to labels 
        (i.e., all dimensions must be either 1, or the same as the corresponding losses dimension).
    label_smoothing: If greater than 0 then smooth the labels.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

Returns:
    Weighted loss Tensor of the same type as logits. If reduction is NONE, this has the same shape as logits; otherwise, it is scalar.
'''

#============================================
#@tf_export('keras.utils.to_categorical')
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
      E.g. for use with categorical_crossentropy.
      Arguments:
          y: class vector to be converted into a matrix
              (integers from 0 to num_classes).
          num_classes: total number of classes.
      Returns:
          A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


#===========================================
sess=tf.InteractiveSession()                        #创建会话
labels=np.array([1,2,3,4,5]).transpose()            #原始标签值
y_=keras.utils.to_categorical(labels,num_classes=10)#二值化后标签值
y=np.random.rand(5,10)                              #模拟预测值
loss=tf.losses.sigmoid_cross_entropy(y_,y)          #损失函数
tf.global_variables_initializer().run()             #初始化变量
print(loss.eval())                                  #计算损失函数
