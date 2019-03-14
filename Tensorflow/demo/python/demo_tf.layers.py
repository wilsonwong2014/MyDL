#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
'''

import os
import sys
import tensorflow as tf
import numpy as np

'''全链接层
tf.layers.dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
Arguments:

    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix. If None (default), weights are initialized using the default initializer used by tf.get_variable.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the kernel after being updated by an Optimizer (e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected variable and must return the projected variable (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the bias after being updated by an Optimizer.
    trainable: Boolean, if True also add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

Returns:

    Output tensor the same shape as inputs except the last dimension is of size units.

范例:
    input_data=tf.placeholder(shape=(None,2),dtype=tf.float32)
    layer1=tf.layers.dense(inputs=input_data,units=3)
    layer1.get_shape()#输出矩阵=>(None,3)
数据表示：
        输入：
        a1_1,a1_2
        a2_1,a2_2
        a3_1,a3_2
        ....
        an_1,an_2
        权重：
        w1_1,w1_2,w1_3
        w2_1,w2_2,w2_3
        输出：
        o1_1=a1_1*w1_1+a1_2*w2_1, o1_2=a1_1*w1_2+a1_2*w2_2, a1_3=a1_1*w1_3+a1_2*w2_3
        o2_1=a2_1*w1_1+a2_2*w2_1, o2_2=a2_1*w1_2+a2_2*w2_2, a2_3=a2_1*w1_3+a2_2*w2_3
        o3_1=a3_1*w1_1+a3_2*w2_1, o3_2=a3_1*w1_2+a3_2*w2_2, a3_3=a3_1*w1_3+a3_2*w2_3
        ............................................................................
        on_1=an_1*w1_1+an_2*w2_1, on_2=an_1*w1_2+an_2*w2_2, an_3=an_1*w1_3+an_2*w2_3
        输出矩阵形状:(n,3)
'''
inputs=tf.placeholder(shape=(None,2),dtype=tf.float32)
layer1=tf.layers.dense(inputs=inputs,units=3) 
print(layer1.get_shape()) #=>(None,3)
layer2=tf.layers.dense(inputs=layer1,units=4)
print(layer2.get_shape()) #=>(None,4)




