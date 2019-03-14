#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Tensorflow可视化 Tensorboard
    https://www.2cto.com/kf/201805/746214.html
    在TensorFlow中，最常用的可视化方法有三种途径，分别为TensorFlow与OpenCv的混合编程、利用Matpltlib进行可视化、利用TensorFlow自带的可视化工具TensorBoard进行可视化。
    这三种方法，在前面博客中都有过比较详细的介绍。但是，TensorFlow中最重要的可视化方法是通过tensorBoard、tf.summary和tf.summary.FileWriter这三个模块相互合作来完成的。

    tf.summary模块的定义位于summary.py文件中，该文件中主要定义了在进行可视化将要用到的各种函数，tf.summary包含的主要函数如下所示：
        from __future__ import                            absolute_import
        from __future__ import                            pision
        from __future__ import                            print_function
        from google.protobuf import json_format as         _json_format
        from tensorflow.core.framework.summary_pb2 import Summary
        from tensorflow.core.framework.summary_pb2 import SummaryDescription
        from tensorflow.core.util.event_pb2 import        Event
        from tensorflow.core.util.event_pb2 import        SessionLog
        from tensorflow.core.util.event_pb2 import        TaggedRunMetadata
         
        from tensorflow.python.eager import context as    _context
        from tensorflow.python.framework import dtypes as _dtypes
        from tensorflow.python.framework import ops as    _ops
        from tensorflow.python.ops import gen_logging_ops as _gen_logging_ops
        from tensorflow.python.ops import summary_op_util as _summary_op_util
        from tensorflow.python.ops.summary_ops import        tensor_summary
        from tensorflow.python.summary.text_summary import   text_summary as text
        from tensorflow.python.summary.writer.writer import       FileWriter
        from tensorflow.python.summary.writer.writer_cache import FileWriterCache
        from tensorflow.python.util import compat as              _compat
        from tensorflow.python.util.all_util import               remove_undocumented
        from tensorflow.python.util.tf_export import              tf_export

    #========================================================================================================
    #模块说明:
    #       tf.summary中包含的主要函数
    #========================================================================================================
    def scalar(name, tensor, collections=None, family=None)                     
    def image(name, tensor, max_outputs=3, collections=None, family=None)
    def histogram(name, values, collections=None, family=None)
    def audio(name, tensor, sample_rate, max_outputs=3, collections=None,family=None)
    def merge(inputs, collections=None, name=None)
    def merge_all(key=_ops.GraphKeys.SUMMARIES, scope=None)
    def get_summary_description(node_def)

    二 tf.summary模块中常用函数的说明：
    1.tf.summary.scalar函数的说明
    #========================================================================================================
    #函数原型:
    #       def scalar(name, tensor, collections=None, family=None)
    #函数说明：
    #       [1]输出一个含有标量值的Summary protocol buffer，这是一种能够被tensorboard模块解析的【结构化数据格式】
    #       [2]用来显示标量信息
    #       [3]用来可视化标量信息
    #       [4]其实，tensorflow中的所有summmary操作都是对计算图中的某个tensor产生的单个summary protocol buffer，而
    #          summary protocol buffer又是一种能够被tensorboard解析并进行可视化的结构化数据格式
    #       虽然，上面的四种解释可能比较正规，但是我感觉理解起来不太好，所以，我将tf.summary.scalar()函数的功能理解为：
    #       [1]将【计算图】中的【标量数据】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备
    #参数说明：
    #       [1]name  :一个节点的名字，如下图红色矩形框所示
    #       [2]tensor:要可视化的数据、张量
    #主要用途：
    #       一般在画loss曲线和accuary曲线时会用到这个函数。
    #=======================================================================================================
    具体的使用方法如下所示：
    #=======================================================================================================================
    #函数说明：
    #       生成【变量】的监控信息，并将生成的监控信息写入【日志文件】
    #参数说明：
    #       [1]var :需要【监控】和【记录】运行状态的【张量】
    #       [2]name:给出了可视化结果中显示的图表名称
    #=======================================================================================================================
    def variable_summaries(var,name):
        with tf.name_scope('summaries'):
            #【1】通过tf.summary.histogram()
            tf.summary.histogram(name,var)
     
            mean   = tf.reduce_mean(var)
            tf.summary.scalar('mean/'+name,mean)
     
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev/'+name,stddev)


    2.tf.summary.image函数的说明
    #========================================================================================================
    #函数原型:
    #       def image(name, tensor, max_outputs=3, collections=None, family=None)
    #函数说明：
    #       [1]输出一个包含图像的summary,这个图像是通过一个4维张量构建的，这个张量的四个维度如下所示：
    #              [batch_size,height, width, channels]
    #       [2]其中参数channels有三种取值：
    #              [1]1: `tensor` is interpreted as Grayscale,如果为1，那么这个张量被解释为灰度图像
    #              [2]3: `tensor` is interpreted as RGB,如果为3，那么这个张量被解释为RGB彩色图像
    #              [3]4: `tensor` is interpreted as Grayscale,如果为4，那么这个张量被解释为RGBA四通道图像
    #       [3]输入给这个函数的所有图像必须规格一致(长，宽，通道，数据类型)，并且数据类型必须为uint8，即所有的像素值在
    #              [0,255]这个范围
    #       虽然，上面的三种解释可能比较正规，但是我感觉理解起来不太好，所以，我将tf.summary.image()函数的功能理解为：
    #       [1]将【计算图】中的【图像数据】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备
    #
    #参数说明：
    #       [1]name  :一个节点的名字，如下图红色矩形框所示
    #       [2]tensor:要可视化的图像数据，一个四维的张量，元素类型为uint8或者float32，维度为[batch_size, height,
    #                 width, channels]
    #       [3]max_outputs:输出的通道数量，可以结合下面的示例代码进行理解
    #主要用途：
    #       一般用在神经网络中图像的可视化
    #========================================================================================================
    示例代码如下所示：
    def main(argv=None):
        #【1】从磁盘加载数据
        mnist = input_data.read_data_sets('F:/MnistSet/',one_hot=True)
        #【2】定义两个【占位符】，作为【训练样本图片/此块样本作为特征向量存在】和【类别标签】的输入变量，并将这些占位符存在命名空间input中
        with tf.name_scope('input'):
            x  = tf.placeholder('float', [None, 784],name='x-input')
            y_ = tf.placeholder('float', [None, 10], name='y-input')
        #【2】将【输入的特征向量】还原成【图片的像素矩阵】，并通过tf.summary.image函数定义将当前图片信息作为写入日志的操作
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x,[-1,28,28,1])
            tf.summary.image('input',image_shaped_input,10)


    3. tf.summary.histogram函数的说明
    #========================================================================================================
    #函数原型:
    #       def histogram(name, values, collections=None, family=None)
    #函数说明：
    #       [1]用来显示直方图信息
    #       [2]添加一个直方图的summary,它可以用于可视化您的数据的分布情况，关于TensorBoard中直方图更加具体的信息可以在
    #          下面的链接https://www.tensorflow.org/programmers_guide/tensorboard_histograms中获取
    #
    #       虽然，上面的两种解释可能比较正规，但是我感觉理解起来不太好，所以，我将tf.summary.histogram()函数的功能理解为：  
    #       [1]将【计算图】中的【数据的分布/数据直方图】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备
    #参数说明：
    #       [1]name  :一个节点的名字，如下图红色矩形框所示
    #       [2]values:要可视化的数据，可以是任意形状和大小的数据  
    #主要用途：
    #       一般用来显示训练过程中变量的分布情况
    #========================================================================================================
    示例代码如下所示：
    #=======================================================================================================================
    #函数说明：
    #       生成一层全连接层神经网络
    #=======================================================================================================================
    def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
                variable_summaries(weights,layer_name+'/weights')
     
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.constant(0.0,shape=[output_dim]))
                variable_summaries(biases,layer_name+'/biases')
     
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor,weights)+biases
                tf.summary.histogram(layer_name+'/pre_activvations',preactivate)
     
            activations = act(preactivate,name='activation')
            tf.summary.histogram(layer_name+'/activations',activations)
            return activations


    4. tf.summary.函数的说明
    #========================================================================================================
    #函数原型:
    #       def merge_all(key=_ops.GraphKeys.SUMMARIES, scope=None)
    #函数说明：
    #       [1]将之前定义的所有summary整合在一起
    #       [2]和TensorFlow中的其他操作类似，tf.summary.scalar、tf.summary.histogram、tf.summary.image函数也是一个
    #          op，它们在定义的时候，也不会立即执行，需要通过sess.run来明确调用这些函数。因为，在一个程序中定义的写日志操作
    #          比较多，如果一一调用，将会十分麻烦，所以Tensorflow提供了tf.summary.merge_all()函数将所有的summary整理在一
    #          起。在TensorFlow程序执行的时候，只需要运行这一个操作就可以将代码中定义的所有【写日志操作】执行一次，从而将
    #          所有的日志写入【日志文件】。
    #
    #参数说明：
    #       [1]key  : 用于收集summaries的GraphKey，默认的为GraphKeys.SUMMARIES
    #       [2]scope：可选参数
    #========================================================================================================


    5. tf.summary.FileWriter类的说明
    #========================================================================================================
    #类定义原型:
    #       class FileWriter(SummaryToEventTransformer)
    #类说明：
    #      [1]将Summary protocol buffers写入磁盘文件
    #      [2]FileWriter类提供了一种用于在给定目录下创建事件文件的机制，并且将summary数据写入硬盘
    #构造函数：
    #        def __init__(self,logdir,graph=None,max_queue=10,flush_secs=120,graph_def=None,filename_suffix=None):
     
    #参数说明：
    #       [1]self  : 类对象自身
    #       [2]logdir：用于存储【日志文件】的目录
    #       [3]graph : 将要存储的计算图
    #应用示例：
    #       summary_writer = tf.summary.FileWriter(SUMMARY_DIR,sess.graph)：创建一个FileWrite的类对象，并将计算图
    #           写入文件
    #========================================================================================================
    示例代码如下所示：
    merged = tf.summary.merge_all()
    #【8】创建回话Session
    with tf.Session() as sess:
        #【9】实例化一个FileWriter的类对象，并将当前TensoirFlow的计算图写入【日志文件】
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR,sess.graph)
        #【10】Tensorflow中创建的变量，在使用前必须进行初始化，下面这个为初始化函数
        tf.global_variables_initializer().run()
        #【11】开始训练
        for i in range(TRAIN_STEPS):
            xs,ys     = mnist.train.next_batch(BATCH_SIZE)
            #【12】运行训练步骤以及所有的【日志文件生成操作】，得到这次运行的【日志文件】。
            summary,_,acc = sess.run([merged,train_step,accuracy],feed_dict={x:xs,y_:ys})
            print('Accuracy at step %s: %s' % (i, acc))
            #【13】将所有的日志写入文件，TensorFlow程序就可以那这次运行日志文件，进行各种信息的可视化
            summary_writer.add_summary(summary,i)
     
    summary_writer.close()


    6.add_summary函数的说明
    #========================================================================================================
    #函数原型：
    #        def add_summary(self, summary, global_step=None)
    #函数说明:
    #        [1]该函数是tf.summary.FileWriter父类中的成员函数
    #        [2]将一个`Summary` protocol buffer添加到事件文件，写入事件文件
    #参数说明：
    #       [1]self   : 类对象自身
    #       [2]summary：将要写入的summary
    #       [3]graph  : global_step,当前迭代的轮数，需要注意的是，如果没有这个参数，那么scalar的summary将会成为一条直线
    #应用示例：
    #       summary_writer.add_summary(summary,i)
    #========================================================================================================
'''

import os
import sys
import tensorflow as tf
import numpy as np

import pdb  #调试
#pdb.set_trace()

#数据目录
file_name=os.path.basename(os.path.splitext(sys.argv[0])[0])
data_path='%s/data/demo/%s'%(os.getenv('HOME'),file_name)
os.makedirs(data_path) if not os.path.exists(data_path) else None
#日志目录
log_dir='%s/log'%(data_path)
os.makedirs(log_dir) if not os.path.exists(log_dir) else None
print('log_dir:',log_dir)

#创建会话
sess=tf.InteractiveSession()

#构造简单计算
a=tf.Variable([1])
b=tf.Variable([2])
c=a+b

#构造tf.summary.scalar数据
#tf.summary.scalar(
#    name,
#    tensor,
#    collections=None,
#    family=None
#)
scalar1=tf.Variable(0,name='scalar1')   #标量初始值不能有[],赋值由op=scalar1.assign(new_val)设置，并通过op.eval()生效
scalar2=tf.Variable(0,name='scalar2')
tf.summary.scalar('scalar1',scalar1)    #生成标量日志信息
tf.summary.scalar('scalar2',scalar2)    #生成标量日志信息

#构造tf.summary.histogram数据[1维]
#tf.summary.histogram(
#    name,
#    values,
#    collections=None,
#    family=None
#)
hist1_1=tf.Variable(np.random.randn(64),name='hist1_1')
hist1_2=tf.Variable(np.random.randn(128),name='hist1_2')
hist1_3=tf.Variable(np.ones([1,100]),name='hist1_3')
tf.summary.histogram('hist1_1',hist1_1)
tf.summary.histogram('hist1_2',hist1_2)
tf.summary.histogram('hist1_3',hist1_3)
#构造tf.summary.histogram数据[2维]
hist2_1=tf.Variable(np.random.randn(64,128),name='hist2_1')
hist2_2=tf.Variable(np.random.randn(128,512),name='hist2_2')
tf.summary.histogram('hist2_1',hist2_1)
tf.summary.histogram('hist2_2',hist2_2)
#构造tf.summary.histogram数据[3维]
hist3_1=tf.Variable(np.random.randn(64,128,3),name='hist3_1')
hist3_2=tf.Variable(np.random.randn(128,512,3),name='hist3_2')
tf.summary.histogram('hist3_1',hist3_1)
tf.summary.histogram('hist3_2',hist3_2)

#构造tf.summary.image数据[10,64,128,1]
#tf.summary.image(
#    name,
#    tensor,
#    max_outputs=3,
#    collections=None,
#    family=None
#)
img1_1=tf.Variable(np.random.randint(0,256,size=(5,64,128,1 ),dtype=np.uint8),name='img1_1')
img1_2=tf.Variable(np.random.randint(0,256,size=(10,64,128,1),dtype=np.uint8),name='img1_2')
tf.summary.image('img1_1',img1_1)
tf.summary.image('img1_2',img1_2)
#构造tf.summary.image数据[10,64,128,3]
img2_1=tf.Variable(np.random.randint(0,256,size=(5,64,128,3 ),dtype=np.uint8),name='img2_1')
img2_2=tf.Variable(np.random.randint(0,256,size=(10,64,128,3),dtype=np.uint8),name='img2_2')
tf.summary.image('img2_1',img2_1)
tf.summary.image('img2_2',img2_2)
#构造tf.summary.image数据[10,64,128,6]
img3_1=tf.Variable(np.random.randint(0,256,size=(5,64,128,6 ),dtype=np.uint8),name='img3_1')
img3_2=tf.Variable(np.random.randint(0,256,size=(10,64,128,6),dtype=np.uint8),name='img3_2')
#tf.summary.image('img3_1',img3_1)
#tf.summary.image('img3_2',img3_2)

merged = tf.summary.merge_all()         #日志记录合并操作
summary_writer = tf.summary.FileWriter(log_dir,sess.graph)

#初始化
tf.global_variables_initializer().run()

#构造日志数据
steps=100
for i in range(steps):
    #tf.summary.scalar
    scalar1.assign(i%50 ).eval()
    scalar2.assign(i%100).eval()
    #tf.summary.image
    img1_1.assign(np.random.randint(0,256,size=( 5,64,128,1 ),dtype=np.uint8)).eval()
    img1_2.assign(np.random.randint(0,256,size=(10,64,128,1 ),dtype=np.uint8)).eval()
    img2_1.assign(np.random.randint(0,256,size=( 5,64,128,3 ),dtype=np.uint8)).eval()
    img2_2.assign(np.random.randint(0,256,size=(10,64,128,3 ),dtype=np.uint8)).eval()
    #tf.summary.audio
    #tf.summary.text
    #tf.summary.histogram

    summary=sess.run(merged)
    summary_writer.add_summary(summary,i)
    
summary_writer.close()
