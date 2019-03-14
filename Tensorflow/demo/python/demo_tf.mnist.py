#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
#------------------------------
INPUT_NODE = 784    #输入节点数
OUTPUT_NODE = 10    #输出节点数
LAYER1_NODE = 500   #隐层节点数
#------------------------------
BATCH_SIZE = 100                #
LEARNING_RATE_BASE = 0.8        #初始化学习率
LEARNING_RATE_DECAY = 0.99      #学习系数
REGULARIZATION_RATE = 0.0001    #正则化参数
TRAINING_STEPS = 30000          #迭代次数
MOVING_AVERAGE_DECAY = 0.99     #移动平均系数
MODEL_SAVE_PATH="MNIST_model/"  #模型保存路径
MODEL_NAME="mnist_model"        #模型名称


#申请权重变量并加入损失参数集合
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1)) #权重参数
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))                         #正则化
    return weights

#创建网络
def inference(input_tensor, regularizer):
    #第一层全链接
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)                       #权重参数
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0)) #偏置参数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)                              #全链接+激活
    #第二层全链接
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)                      #权重参数
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0)) #偏置参数
        layer2 = tf.matmul(layer1, weights) + biases                                                #全链接
    return layer2


#网络训练
def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')    #网络输入
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')  #网络输出

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE) #正则化
    y = inference(x, regularizer)                                       #创建网络
    global_step = tf.Variable(0, trainable=False)                       #全局迭代次数变量

    #滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #指数学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    #优化方法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver() #持久化保存

    #网络训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()     #全局变量初始化

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE) #批数据
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()


'''网络评估
'''
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train


# #### 1. 每10秒加载一次最新的模型

# In[2]:


# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

#网络评估
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')   #网络输入
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input') #网络输出
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}            #feed_dict

        y = inference(x, None)                                              #创建网络
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


# ###  主程序

# In[3]:


def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()


