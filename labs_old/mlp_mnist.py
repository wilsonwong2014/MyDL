#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''mlp网络实验
    网络模型：mlp
    数 据 集：mnist
    实验结果：
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        dense_1 (Dense)              (None, 512)               401920    
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 512)               0         
        _________________________________________________________________
        dense_2 (Dense)              (None, 512)               262656    
        _________________________________________________________________
        dropout_2 (Dropout)          (None, 512)               0         
        _________________________________________________________________
        dense_3 (Dense)              (None, 10)                5130      
        =================================================================
        Total params: 669,706
        Trainable params: 669,706
        Non-trainable params: 0
        _________________________________________________________________
        2018-10-27 14:29:50.997004: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
        2018-10-27 14:29:51.162061: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
        name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
        pciBusID: 0000:65:00.0
        totalMemory: 10.91GiB freeMemory: 10.49GiB
        2018-10-27 14:29:51.162095: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
        2018-10-27 14:29:51.374813: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
        2018-10-27 14:29:51.374856: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
        2018-10-27 14:29:51.374865: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
        2018-10-27 14:29:51.375074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10140 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:65:00.0, compute capability: 6.1)
        Epoch 1/20
        60000/60000 [==============================] - 2s 38us/step - loss: 1.2016 - acc: 0.6852
        Epoch 2/20
        60000/60000 [==============================] - 2s 35us/step - loss: 0.5263 - acc: 0.8494
        Epoch 3/20
        60000/60000 [==============================] - 2s 35us/step - loss: 0.4210 - acc: 0.8778
        ......
        Epoch 18/20
        60000/60000 [==============================] - 2s 35us/step - loss: 0.1840 - acc: 0.9471
        Epoch 19/20
        60000/60000 [==============================] - 2s 35us/step - loss: 0.1791 - acc: 0.9477
        Epoch 20/20
        60000/60000 [==============================] - 2s 35us/step - loss: 0.1727 - acc: 0.9498

    使用范例:
        #数据
            mnist.load_data()
        #训练
            $python3 LeNet_mnist.py --datapath ~/data/mnist --testpath test
        #预测
            $python3 LeNet_mnist.py --fun 1 --datapath ~/data/mnist --testpath test
        #调试
            加参数 --dbg 1
        #Tensorboard
            日志目录:~/data/mnist/LeNet_logdir
            $tensorboard --log_dir=~/data/mnist/LeNet_logdir
            浏览器访问：http://localhost:6006
'''
from __future__ import print_function

import os
import sys
import pdb

import keras
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import argparse
import cv2

#mlp训练mnist数据集
class mlp_mnist(object):
    '''使用范例:
    obj=mlp_mnist('~/data/mlp_mnist','test') #声明对象
    obj.LoadData()                           #加载数据
    obj.Train()                              #训练
    obj.Evaluate()                           #评估
    obj.Predicts()                           #预测
    obj.Report()                             #实验报告
    '''
    #类似构造函数
    def __init__(self,data_path,test_path,log_dir,model_file,result_file,epochs):
        '''
         @param data_path   数据目录
         @param test_path   测试目录
         @param log_dir     日志目录
         @param model_file  模型参数文件
         @param result_file 预测结果文件
         @param epochs      迭代次数
        '''
        #-------通用参数-------
        self.data_path=data_path                #数据输出目录
        self.model_file=model_file              #模型参数文件
        self.log_dir=log_dir                    #输出日志
        #self.train_dir='%s/train'%(data_path)  #训练目录
        #self.valid_dir='%s/valid'%(data_path)  #校验目录
        self.test_dir=test_path                 #测试目录
        self.log_dir=log_dir                    #日志目录
        self.result_file=result_file            #预测结果文件
        self.validation_split=0.2               #从训练数据中划分一定比例用于验证,0.0-1.0
        self.batch_size=128                     #批大小
        self.epochs=epochs                      #迭代次数
        #-------私有参数-----------
        self.model=None      #模型对象
        self.img_rows=28     #图像行数
        self.img_cols=28     #图像列数
        self.img_chs=1       #图像通道数   
        self.input_shape=(self.img_rows*self.img_cols,)  #输入维度
        self.train_num=60000 #训练样本数
        self.test_num=10000  #测试样本数
        self.num_classes=10  #类别数
        
        self.x_train=None    #训练输入数据,[?,784]
        self.y_train=None    #训练标签数据,[?,784]
        self.x_test=None     #测试输入数据,[?,10]
        self.y_test=None     #测试标签数据,[?,10]
        self.histroy=None    #训练状态,tf.keras.callbacks.History ,{'loss': [1.2449509007612864, ...，迭代次数], 'acc': [0.6591,...，迭代次数]}
        self.score=None      #评估得分(loss,acc)
        #-------目录检测并自动创建-------
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(os.path.dirname(self.result_file)):
            os.makedirs(os.path.dirname(self.result_file))

    #加载数据
    def LoadData(self):
        ''' 数据加载
        '''
        # the data, split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.reshape(self.train_num, self.img_rows*self.img_cols)
        self.x_test = self.x_test.reshape(self.test_num, self.img_rows*self.img_cols)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)


    #创建模型
    def CreateModel(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        self.model=model 


    #加载模型 
    def LoadModel(self):
        if os.path.exists(self.model_file):
            model = load_model(self.model_file)
        else:
            model=None
        self.model=model


    #模型训练
    def Train(self):
        ''' 训练
        '''
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        self.history = self.model.fit(self.x_train, self.y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    callbacks=[TensorBoard(log_dir=self.log_dir)])
        #保存模型参数
        self.model.save(self.model_file)


    #模型评估
    def Evaluate(self):
        ''' 评估
        @return score:得分(loss,acc)
        '''
        self.score = self.model.evaluate(self.x_test, self.y_test, verbose=0)


    #模型预测
    def Predict(self,x):
        ''' 测试
        @param x:测试数据
        @return y:测试结果
        '''
        return self.model.predict(x)
    

    #-------------------------------
    def get_img(self,sfile):
        '''图像读取并预处理:28x28
        @return img:返回[1,28x28]行向量
        '''
        img=cv2.imread(sfile,cv2.IMREAD_GRAYSCALE) #读取图像
        if isinstance(img,np.ndarray):
            #图像预处理 ...
            img.resize((self.img_rows,self.img_cols))            #重置图像大小
            return img.reshape(1,self.img_rows*self.img_cols)  #返回行向量
        else:
            return None


    #-------------------------------
    def Predicts(self):
        '''模型预测,每一个文件都是一个单独的数字
           加载模型文件=>遍历预测目录，读取图形文件=>图像预测=>存放预测结果
        '''
        testpath=self.test_dir              #测试目录，不支持递归遍历
        savefile=self.result_file           #测试结果保存文件
        if not os.path.exists(self.model_file):
            print('model_file:%s not exists!' %(model_file))
            return
        if not os.path.exists(testpath):
            print('testpath:%s not exists!' %(testpath))
            return
        #检索图像文件列表
        files=os.listdir(testpath)
        with open(savefile,'w') as f:
            for s in files:
                sfile='%s/%s' %(testpath,s)
                x=self.get_img(sfile)
                if isinstance(x,np.ndarray):
                    y=self.Predict(x)
                    y_index=np.where(y==y.max())
                    f.write('%s:%d:%s\r\n'%(s,y_index[1][0],y))
                    print('file:%s=>%d',sfile,y_index[1][0])
                else:
                    print('get_img(%s):=> fail!'%(sfile))
    

    #实验报告
    def Report(self,**kwargs):
        pass


#end class LeNet_mnist(object)
#==============================


def params():
    ''' 程序参数
    '''
    #程序描述
    description=''
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--fun',type=int, help='脚本功能[0-训练,1-预测],默认0. eg. --fun 0',default=0);
    parser.add_argument('--datapath', type=str, help='数据目录. eg. --datapath "~/data/mnist"', default='%s/data/mnist'%(os.getenv('HOME')));
    parser.add_argument('--testpath', type=str, help='测试目录. eg. --testpath "test"', default='test');
    parser.add_argument('--model', type=str, help='参数文件. eg. --model "mlp.h5"', default='mlp.h5');
    parser.add_argument('--log_dir', type=str, help='日志目录. eg. --log_dir "mlp_logdir"', default='mlp_logdir');
    parser.add_argument('--result', type=str, help='预测结果. eg. --result "mlp_result"', default='mlp_result');
    parser.add_argument('--epochs', type=int, help='迭代次数. eg. --epochs 20', default=20);
    parser.add_argument('--dbg', type=int, help='是否调试. eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    #--------------------------
    return arg


#-------------------------------
def main(arg):
    ''' 主函数
    '''
    datapath=arg.datapath
    testpath=arg.testpath
    model_file=arg.model
    log_dir=arg.log_dir
    result_file=arg.result
    if testpath[0]!='/':
        testpath='%s/%s'%(datapath,testpath)
    if model_file[0]!='/':
        model_file='%s/%s'%(datapath,model_file)
    if log_dir[0]!='/':
        log_dir='%s/%s'%(datapath,log_dir)
    if result_file[0]!='/':
        result_file='%s/%s'%(testpath,result_file)
    #----------------------------------------------
    obj=mlp_mnist(datapath,testpath,log_dir,model_file,result_file,arg.epochs)
    if arg.fun==0:
        obj.LoadData()          #加载数据
        obj.CreateModel()       #创建模型
        obj.Train()             #训练
        score=obj.Evaluate()    #评估
        obj.Predicts()          #预测
        obj.Report()            #实验报告
    else:
        obj.LoadModel()         #加载模型
        obj.Predicts()          #预测


#==============================
if __name__=='__main__':
    arg=params() #命令行参数解析
    main(arg)
