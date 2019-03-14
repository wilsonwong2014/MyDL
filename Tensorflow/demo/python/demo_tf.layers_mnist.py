#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''多层神经网络范例
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

data :random,mnist,cifar10
model:tf,tf.layers,tf.keras
'''
import os
import sys
import tensorflow as tf
from numpy.random import RandomState

class Demo_Net(object):
    #构造函数
    def __init__(self,logdir):
        self.logdir=logdir  #日志目录
        #模拟数据参数
        self.num_class=10                             #分类个数
        self.num_feature=3                            #特征数
        self.data_std=0.1                             #模拟数据标准差
        self.train_num_class=5000                     #每类训练样本数
        self.valid_num_class=1000                     #每类校验样本数
        self.test_num_class=1000                      #每类测试样本数
        #网络层参数
        num_layers=[10,10]  #网络层数目
        #损失函数参数
        # ......
        #训练参数
        self.epochs=10      #迭代次数
        self.batch_size=32  #

    #-----------------------数据加载---------------------
    #加载数据
    def LoadData(self):
        '''随机构造训练数据
            正态分布：mean=[0,1,2,3,4,5,6,7,8,9],std=0.1,samples=5000/class
        @return (x_train,y_train,x_valid,y_valid,x_test,y_test)
        '''
        #参数设置
        num_class=self.num_class                 #分类个数
        num_feature=self.num_feature             #特征数
        data_std=self.data_std                   #模拟数据标准差
        train_num_class=self.train_num_class     #每类训练样本数
        valid_num_class=self.valid_num_class     #每类校验样本数
        test_num_class=self.test_num_class       #每类测试样本数
        #样本集
        labels=[i for i in range(num_class)]     #标签[均值]
        datas_train=np.zeros([0,num_feature+1])  #训练样本集
        datas_valid=np.zeros([0,num_feature+1])  #校验样本集
        datas_test =np.zeros([0,num_feature+1])  #测试样本集
        #数据生成
        for i in range(num_class):
            data_train=np.random.normal(i,std,size=(train_num_class,num_feature+1)) #最后一列存放标签
            data_train[:,-1]=i                                                      #设置标签值
            datas_train=np.vstack(datas_train,data_train)                           #垂直拼接
            data_valid=np.random.normal(i,std,size=(valid_num_class,num_feature+1)) #最后一列存放标签
            data_valid[:,-1]=i                                                      #设置标签值
            datas_valid=np.vstack(datas_valid,data_valid)                           #垂直拼接
            data_test=np.random.normal(i,std,size=(test_num_class,num_feature+1))   #最后一列存放标签
            data_test[:,-1]=i                                                       #设置标签值
            datas_test=np.vstack(datas_test,data_test)                              #垂直拼接    
        #数据打乱
        np.random.shuffle(datas_train) #数据打乱
        np.random.shuffle(datas_valid) #数据打乱
        np.random.shuffle(datas_test)  #数据打乱
        return (datas_train[:,:-1],datas_train[:,-1],datas_valid[:,:-1],datas_valid[:,-1],datas_test[:,:-1],datas_test[:,-1])
        
    #------------------网络层------------------
    #全链接模型{1}:通过tf.layers.dense创建
    def Dense_1(self,units,inputs=None):
        layer=tf.layers.dense(
                inputs=inputs,
                units=units,
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
        return layer
    
    #-----------------模型方案------------------
    #构建模型:方案1
    def Model_1(self,x):
        num_layers=self.num_layers  #网络层数目
        cur_layer=self.Dense_1(inputs=x,units=num_layers[0])
        for n in num_layers[1:]:
            cur_layer=self.Dense_1(inputs=cur_layer,units=n)
        return cur_layer

    #-----------------损失函数------------------
    #损失函数:方案1
    def Losses_1(self,y,y_):
        loss=tf.losses.sigmoid_cross_entropy(y_,y)          #损失函数
        return loss        

    #----------------训练方案-------------------
    #模型训练:方案1
    def Train_1(self):
        # #### 1. 定义神经网络的参数，输入和输出节点。
        batch_size = 8
        w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
        w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
        x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
        y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
        # #### 2. 定义前向传播过程，损失函数及反向传播算法。
        a = tf.matmul(x, w1)
        y = tf.matmul(a, w2)
        y = tf.sigmoid(y)
        cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                        + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
        # ####  3. 生成模拟数据集。
        rdm = RandomState(1)
        X = rdm.rand(128,2)
        Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
        # #### 4. 创建一个会话来运行TensorFlow程序。
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # 输出目前（未经训练）的参数取值。
            print(sess.run(w1))
            print(sess.run(w2))
            print("\n")
            # 训练模型。
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
            print(sess.run(w1))
            print(sess.run(w2))

   
    #-------------模型评估----------------------
    #模型评估:方案1
    def Evaluate_1(self):
        pass

    #-------------模型预测----------------------
    #模型预测:方案1
    def Predict_1(self):
        pass

    #-------------综合方案----------------------
    #构建模型
    def ModelCreate(self,x):
        self.ModelCreate_1(x)

    #损失函数
    def ModelLosses(self,y,y_):
        loss=self.Losses_1(y_,y)          #损失函数
        return loss

    #模型训练
    def ModelTrain(self):
        pass

    #模型评估
    def ModelEvaluate(self):
        pass

    #模型测试
    def ModelPredict(self):
        pass

# #### 1. 定义神经网络的参数，输入和输出节点。

# In[2]:


batch_size = 8
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')


# #### 2. 定义前向传播过程，损失函数及反向传播算法。

# In[3]:


a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


# ####  3. 生成模拟数据集。

# In[4]:


rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


# #### 4. 创建一个会话来运行TensorFlow程序。

# In[5]:


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    # 输出目前（未经训练）的参数取值。
    print(sess.run(w1))
    print(sess.run(w2))
    print("\n")
    
    # 训练模型。
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
    print(sess.run(w1))
    print(sess.run(w2))

