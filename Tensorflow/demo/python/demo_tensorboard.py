#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
本代码块有异常！
'''

# In[1]: 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 

import keras
from keras.datasets import mnist
# ####  1. 生成变量监控信息并定义生成监控信息日志的操作。 
# In[2]: 
SUMMARY_DIR = "log2" 
BATCH_SIZE = 100 
TRAIN_STEPS = 3000 

def variable_summaries(var, name): 
    with tf.name_scope('summaries'): 
        tf.summary.histogram(name, var) 
        mean = tf.reduce_mean(var) 
        tf.summary.scalar('mean/' + name, mean) 
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean))) 
        tf.summary.scalar('stddev/' + name, stddev) 

# #### 2. 生成一层全链接的神经网络。 
# In[3]: 
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu): 
    with tf.name_scope(layer_name): 
        with tf.name_scope('weights'): 
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1)) 
            variable_summaries(weights, layer_name + '/weights') 
        with tf.name_scope('biases'): 
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim])) 
            variable_summaries(biases, layer_name + '/biases') 
        with tf.name_scope('Wx_plus_b'): 
            preactivate = tf.matmul(input_tensor, weights) + biases 
            tf.summary.histogram(layer_name + '/pre_activations', preactivate) 
        activations = act(preactivate, name='activation') 
        
        # 记录神经网络节点输出在经过激活函数之后的分布。 
        tf.summary.histogram(layer_name + '/activations', activations) 
        return activations 

# In[4]: 
def main(): 
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True) 
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #y_train = keras.utils.to_categorical(y_train, 10)
    #y_test = keras.utils.to_categorical(y_test, 10)
    with tf.name_scope('input'): 
        x = tf.placeholder(tf.float32, [None, 784], name='x-input') 
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input') 
    with tf.name_scope('input_reshape'): 
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1]) 
        tf.summary.image('input', image_shaped_input, 10) 

    hidden1 = nn_layer(x, 784, 500, 'layer1') 
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity) 

    with tf.name_scope('cross_entropy'): 
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)) 
        tf.summary.scalar('cross_entropy', cross_entropy) 

    with tf.name_scope('train'): 
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy) 

    with tf.name_scope('accuracy'): 
        with tf.name_scope('correct_prediction'): 
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
        with tf.name_scope('accuracy'): 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        tf.summary.scalar('accuracy', accuracy) 
    merged = tf.summary.merge_all() 
    with tf.Session() as sess: 
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph) 
        tf.global_variables_initializer().run() 
        for i in range(TRAIN_STEPS): 
            xs, ys = mnist.train.next_batch(BATCH_SIZE) 
            #xs=x_train
            #ys=y_train
            # 运行训练步骤以及所有的日志生成操作，得到这次运行的日志。 
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys}) 
            # 将得到的所有日志写入日志文件，这样TensorBoard程序就可以拿到这次运行所对应的 
            # 运行信息。 
            summary_writer.add_summary(summary, i) 
    summary_writer.close() 

# In[5]: 

if __name__ == '__main__': 
    main()

'''
--------------------- 
作者：GoHust_Liu 
来源：CSDN 
原文：https://blog.csdn.net/qq_33039859/article/details/80277379 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''
