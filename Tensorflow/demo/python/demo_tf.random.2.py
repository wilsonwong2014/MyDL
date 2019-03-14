#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
from numpy.random import RandomState

# #### 1. 定义神经网络的参数，输入和输出节点。
batch_size = 8
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))     #第一层全链接权重
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))     #第二层全链接权重
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input") #输入层
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input') #输出层

# #### 2. 定义前向传播过程，损失函数及反向传播算法。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)
#损失函数，交叉熵
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                        + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
#优化算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# ####  3. 生成模拟数据集。
rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X] # 128 x 1
print('X.shape:',X.shape)
print('type(Y):',type(Y))

# #### 4. 创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print('训练前参数:')
    print('w1:\n',sess.run(w1))
    print('w2:\n',sess.run(w2))
    print("\n")
    
    # 训练模型。
    print('开始训练')
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    
    # 输出训练后的参数取值。
    print("\n")
    print('训练后参数:')
    print('w1:\n',sess.run(w1))
    print('w2:\n',sess.run(w2))

