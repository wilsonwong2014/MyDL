#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#1. 图，张量及会话.ipynb
###########################
#计算图定义
import tensorflow as tf
#计算图1
g1 = tf.Graph();
with g1.as_default():
    #定义变量并设置初值,声明变量范围:with tf.variable_scope("scope_name"):
    v = tf.get_variable("v", [1], initializer = tf.zeros_initializer()) # 标量,设置初始值为0
#计算图2
g2 = tf.Graph();
with g2.as_default():
    #定义变量并设置初值
    v = tf.get_variable("v", [1], initializer = tf.ones_initializer())  # 标量,设置初始值为1

###########################
#会话定义
#会话设置计算图1    
with tf.Session(graph = g1) as sess:
    #初始化所有变量,xxxx.run()相当于sess.run(xxxx);
    tf.global_variables_initializer().run();
    with tf.variable_scope("", reuse=True):
        print("value of g1.v=>");
        print(sess.run(tf.get_variable("v")));
        '''
        输出: 0
        '''
#会话设置计算图2
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run();
    with tf.variable_scope("", reuse=True):
        print("value of g2.v=>");
        print(sess.run(tf.get_variable("v")));
        '''
        输出: 1
        '''

##########################
#张量概念
# #### 2. 张量的概念
# In[2]:
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a");
b = tf.constant([2.0, 3.0], name="b");
result = a + b;
print("Tensor:result=a+b=>");
print (result);
'''
输出:Tensor("add:0", shape=(2,), dtype=float32)

'''

sess = tf.InteractiveSession ();
print("value of result=>");
print(result.eval());
'''
输出:[3. 5.]
'''
sess.close();

########################
# #### 3. 会话的使用
# 3.1 创建和关闭会话
# In[3]:
# 创建一个会话。
sess = tf.Session();

# 使用会话得到之前计算的结果。
print("value of result");
print(sess.run(result));
'''
输出:[3. 5.]
'''
# 关闭会话使得本次运行中使用到的资源可以被释放。
sess.close()


# 3.2 使用with statement 来创建会话
# In[4]:
with tf.Session() as sess:
    print("value of result=>");
    print(sess.run(result));
    '''
    输出:[3. 5.]
    '''
# 3.3 指定默认会话
# In[5]:
sess = tf.Session();
with sess.as_default():
    print("value of result=>");
    print(result.eval());
    '''
    输出:[3. 5.]
    '''
# In[6]:
sess = tf.Session();

# 下面的两个命令有相同的功能。
print("value of result=>");
print(sess.run(result));
'''
输出:[3. 5.]
'''
print("value of result=>");
print(result.eval(session=sess))
'''
输出:[3. 5.]
'''
# #### 4. 使用tf.InteractiveSession构建会话
# In[7]:
sess = tf.InteractiveSession ();
print("value of result=>");
print(result.eval());
'''
输出:[3. 5.]
'''
sess.close();

# #### 5. 通过ConfigProto配置会话
# In[8]:
config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False);
sess1 = tf.InteractiveSession(config=config);
sess2 = tf.Session(config=config);

