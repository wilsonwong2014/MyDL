#!/usr/bin/env  python3
# -*- coding: utf-8 -*-
'''
--------------------- 
作者：C小C 
来源：CSDN 
原文：https://blog.csdn.net/C_chuxin/article/details/85269471 
版权声明：本文为博主原创文章，转载请附上博文链接！

tmp1:
1.1425604
tmp2:
1.8767173
y:
[1.1425604 1.8767173]
x:
[[1.1425604  0.93835866]]
the gradient of y=x[0,0]+2*x[0,1] for x is : [array([[1., 2.]], dtype=float32)]

'''
import tensorflow as tf
 
x = tf.get_variable('w1',shape=[1,2])
 
tmp1=x[0,0]
tmp2=2*x[0,1]
y=tf.stack([tmp1,tmp2],0)  #y 是一个（2,1）的张量，x 是一个（1,2）的张量,而且y=x[0,0]+2*x[0,1]
 
grads = tf.gradients(y,[x])
 
with tf.Session() as sess:
 
    tf.global_variables_initializer().run()
 
    re = sess.run(grads)
 
    print('tmp1:')
    print(tmp1.eval())
    print('tmp2:')
    print(tmp2.eval())
    print('y:')
    print(y.eval())
    print('x:')
    print(x.eval())

    print('the gradient of y=x[0,0]+2*x[0,1] for x is :',re)


