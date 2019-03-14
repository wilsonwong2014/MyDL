#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
tf.Variable:
__init__(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None,
    constraint=None
)

'''

import tensorflow as tf

sess=tf.InteractiveSession()
v=tf.Variable(initial_value=tf.random_normal((2,3),mean=0.0,stddev=1.0),name='v')

#初始化单个变量
#sess.run(v.initializer) 

#初始化所有变量
init = tf.global_variables_initializer()
sess.run(init)

print(v)
print(v.eval())
