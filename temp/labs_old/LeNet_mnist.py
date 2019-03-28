#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''LeNet网络实验
    网络模型：LeNet
    数 据 集：mnist
    实验结果：
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d_1 (Conv2D)            (None, 24, 24, 32)        832       
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 8, 8, 64)          51264     
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 1024)              0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 100)               102500    
        _________________________________________________________________
        dense_2 (Dense)              (None, 10)                1010      
        =================================================================
        Total params: 155,606
        Trainable params: 155,606
        Non-trainable params: 0

        Epoch 1/20
        60000/60000 [==============================] - 3s 54us/step - loss: 1.2267 - acc: 0.6524
        Epoch 2/20
        60000/60000 [==============================] - 3s 43us/step - loss: 0.2957 - acc: 0.9114
        Epoch 3/20
        60000/60000 [==============================] - 3s 43us/step - loss: 0.2034 - acc: 0.9392
        .....
        Epoch 18/20
        60000/60000 [==============================] - 3s 43us/step - loss: 0.0504 - acc: 0.9846
        Epoch 19/20
        60000/60000 [==============================] - 3s 42us/step - loss: 0.0482 - acc: 0.9851
        Epoch 20/20
        60000/60000 [==============================] - 3s 43us/step - loss: 0.0461 - acc: 0.9861

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
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
import numpy as np
import argparse
import cv2


#LeNet训练mnist数据集
class LeNet_mnist(object):
    '''使用范例:
    obj=LeNet_mnist('~/data/LeNet_mnist','test')   #声明对象
    obj.LoadData()                                 #加载数据
    his=obj.Train()                                #训练
    score=obj.Evaluate()                           #评估
    y_predict=obj.Predict()                        #预测
    obj.Report()                                   #实验报告
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
        self.data_path=data_path                    #数据输出目录
        self.model_file=model_file                  #模型参数文件
        self.log_dir=log_dir                        #输出日志
        #self.train_dir='%s/train'%(data_path)      #训练目录
        #self.valid_dir='%s/valid'%(data_path)      #校验目录
        self.test_dir=test_path                     #测试目录
        self.log_dir=log_dir                        #日志目录
        self.result_file=result_file                #预测结果文件
        self.validation_split=0.2                   #从训练数据中划分一定比例用于验证,0.0-1.0
        self.batch_size=128                         #批大小
        self.epochs=epochs                          #迭代次数
        #-------私有参数-----------
        self.model=None      #模型对象
        self.img_rows=28     #图像行数
        self.img_cols=28     #图像列数
        self.img_chs=1       #图像通道数   
        self.input_shape=(self.img_rows,self.img_cols,self.img_chs)  #输入维度
        self.train_num=60000 #训练样本数
        self.test_num=10000  #测试样本数
        self.num_classes=10  #类别数
        
        self.x_train=None    #训练输入数据,[?,28,28,1]
        self.y_train=None    #训练标签数据,[?,28,28,1]
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
        self.x_train = self.x_train.reshape(self.train_num, self.img_rows, self.img_cols,self.img_chs)
        self.x_test = self.x_test.reshape(self.test_num, self.img_rows, self.img_cols,self.img_chs)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)


    #创建模型
    def CreateModel(self):
        '''
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d_1 (Conv2D)            (None, 24, 24, 32)        832       
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 8, 8, 64)          51264     
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 1024)              0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 100)               102500    
        _________________________________________________________________
        dense_2 (Dense)              (None, 10)                1010      
        =================================================================
        Total params: 155,606
        Trainable params: 155,606
        Non-trainable params: 0
        '''
        model = Sequential()  
        model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=self.input_shape,padding='valid',activation='relu',kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Flatten())  
        model.add(Dense(100,activation='relu'))  
        model.add(Dense(self.num_classes,activation='softmax'))
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
        @return img:返回[1,28,28,1]
        '''
        img=cv2.imread(sfile,cv2.IMREAD_GRAYSCALE) #读取图像
        if isinstance(img,np.ndarray):
            #图像预处理 ...
            img.resize((self.img_rows,self.img_cols))            #重置图像大小
            return img.reshape(1,self.img_rows,self.img_cols,self.img_chs)  #返回行向量
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
        '''
        '''
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
    parser.add_argument('--model', type=str, help='参数文件. eg. --model "LeNet.h5"', default='LeNet.h5');
    parser.add_argument('--log_dir', type=str, help='日志目录. eg. --log_dir "LeNet_logdir"', default='LeNet_logdir');
    parser.add_argument('--result', type=str, help='预测结果. eg. --result "LeNet_result"', default='LeNet_result');
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
    obj=LeNet_mnist(datapath,testpath,log_dir,model_file,result_file,arg.epochs)
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

