#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''tf.constant范例
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
)
'''

import tensorflow as tf
sess=tf.InteractiveSession()    #执行变量值，必须创建session
v1=tf.constant(1)
v2=v1
print('type(v1):')
print(type(v1))
print('v1:')
print(v1)
print('v1.eval():')
print(v1.eval())

#tf.constant是否可以修改？不可以

