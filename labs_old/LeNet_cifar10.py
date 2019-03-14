#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''LeNet网络实验
    How to Check-Point Deep Learning Models in Keras
        https://machinelearningmastery.com/check-point-deep-learning-models-keras/
        https://keras-cn.readthedocs.io/en/latest/other/callbacks/#modelcheckpoint
    网络模型：LeNet
    数 据 集：cirfar10
    实验结果：
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d_1 (Conv2D)            (None, 28, 28, 32)        2432      
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 10, 10, 64)        51264     
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 1600)              0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 100)               160100    
        _________________________________________________________________
        dense_2 (Dense)              (None, 10)                1010      
        =================================================================
        Total params: 214,806
        Trainable params: 214,806
        Non-trainable params: 0
        _________________________________________________________________
        2018-10-29 10:06:11.147423: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
        2018-10-29 10:06:11.269259: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
        name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
        pciBusID: 0000:65:00.0
        totalMemory: 10.91GiB freeMemory: 10.54GiB
        2018-10-29 10:06:11.269291: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
        2018-10-29 10:06:11.478231: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
        2018-10-29 10:06:11.478270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
        2018-10-29 10:06:11.478275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
        2018-10-29 10:06:11.478483: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10196 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:65:00.0, compute capability: 6.1)
        Epoch 1/200
        50000/50000 [==============================] - 4s 84us/step - loss: 0.2882 - acc: 0.9065
        Epoch 2/200
        50000/50000 [==============================] - 4s 70us/step - loss: 0.2843 - acc: 0.9066
        Epoch 3/200
        50000/50000 [==============================] - 4s 71us/step - loss: 0.2750 - acc: 0.9118
        ......
        Epoch 73/200
        50000/50000 [==============================] - 3s 69us/step - loss: 0.0132 - acc: 1.0000
        Epoch 74/200
        50000/50000 [==============================] - 4s 71us/step - loss: 0.0127 - acc: 1.0000
        Epoch 75/200
        50000/50000 [==============================] - 3s 68us/step - loss: 0.0123 - acc: 0.9999
        Epoch 76/200
        50000/50000 [==============================] - 3s 68us/step - loss: 0.0120 - acc: 0.9999
        ------------
        Predicts{img_count:50,acc_count:28,acc:0.560000}

    使用范例:
        #数据
            cifar10.load_data()
        #训练
            $python3 LeNet_cifar10.py --datapath ~/data/cifar10 --testpath test
        #预测
            $python3 LeNet_cifar10.py --fun 1 --datapath ~/data/cifar10 --testpath test
        #调试
            加参数 --dbg 1
        #Tensorboard
            日志目录:~/data/cifar10/LeNet_logdir
            $tensorboard --log_dir=~/data/cifar10/LeNet_logdir
            浏览器访问：http://localhost:6006
'''
from __future__ import print_function

import os
import sys
import pdb

import keras
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
import numpy as np
import argparse
import cv2


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


#LeNet训练mnist数据集
class LeNet_mnist(object):
    '''使用范例:
    obj=LeNet_mnist('~/data/cifar10','test','LeNet_logdir','LeNet.h5','LeNet_result.txt',20)   #声明对象
    obj.LoadData()                                 #加载数据
    his=obj.Train()                                #训练
    score=obj.Evaluate()                           #评估
    obj.Predicts()                                 #预测
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
        self.img_rows=32     #图像行数
        self.img_cols=32     #图像列数
        self.img_chs=3       #图像通道数   
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
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        #样本参数修正 begin
        self.train_num=self.x_train.shape[0]
        self.test_num=self.x_test.shape[0]
        self.img_rows=self.x_train.shape[1]
        self.img_cols=self.x_train.shape[2]
        self.img_chs=self.x_train.shape[3]
        self.input_shape=self.x_train.shape[1:]
        #样本参数修正 end       
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
        #输出模型概况
        self.model.summary()
        #训练断点加载
        #filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        cp_file='%s_checkpoint.h5'%(os.path.splitext(self.model_file)[0])
        if os.path.exists(cp_file):
            self.model.load_weights(cp_file)
        #模型编译
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        #断点训练:monitor监控参数可以通过self.score = self.model.evaluate(self.x_test, self.y_test, verbose=0)的score查询
        checkpoint_cb = ModelCheckpoint(cp_file, monitor='acc', verbose=1, save_best_only=True, mode='auto',period=10)
        #EarlyStopping
        earlyStopping_cb=keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0, mode='max')
        #TensorBoard
        #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        #参数：
        #   log_dir：保存日志文件的地址，该文件将被TensorBoard解析以用于可视化
        #   histogram_freq：计算各个层激活值直方图的频率（每多少个epoch计算一次），如果设置为0则不计算。
        #   write_graph: 是否在Tensorboard上可视化图，当设为True时，log文件可能会很大
        #   write_images: 是否将模型权重以图片的形式可视化
        #   embeddings_freq: 依据该频率(以epoch为单位)筛选保存的embedding层
        #   embeddings_layer_names:要观察的层名称的列表，若设置为None或空列表，则所有embedding层都将被观察。
        #   embeddings_metadata: 字典，将层名称映射为包含该embedding层元数据的文件名，参考这里获得元数据文件格式的细节。如果所有的embedding层都使用相同的元数据文件，则可传递字符串。
        tensorBoard_cb=TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=2, embeddings_layer_names=None, embeddings_metadata=None)
        #callbacks = [TensorBoardWrapper(gen_val, nb_steps=5, log_dir=self.cfg['cpdir'], histogram_freq=1, batch_size=32, write_graph=False, write_grads=True)]
        #回调函数序列
        callbacks_list = [checkpoint_cb,earlyStopping_cb,tensorBoard_cb]
        #模型训练
        self.history = self.model.fit(self.x_train, self.y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    callbacks=callbacks_list)
        #保存模型参数
        self.model.save(self.model_file)


    #模型评估
    def Evaluate(self):
        ''' 评估
        @return score:得分(loss,acc)
        '''
        self.score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('score{loss:%s,acc:%s}'%(self.score[0],self.score[1]))


    #模型预测
    def Predict(self,x):
        ''' 测试
        @param x:测试数据
        @return y:测试结果
        '''
        return self.model.predict(x)
    

    #-------------------------------
    def get_img(self,sfile):
        '''图像读取并预处理:[32,32,3]
        @return img:返回[1,32,32,3]
        '''
        img=cv2.imread(sfile) #读取图像
        if isinstance(img,np.ndarray):
            #图像预处理 ...
            #img.resize((self.img_rows,self.img_cols))            #重置图像大小
            #print(img.shape)
            img=cv2.resize(img,(self.img_rows,self.img_cols))
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
        img_count=0     #图像个数统计
        acc_count=0     #预测正确统计
        with open(savefile,'w') as f:
            for s in files:
                sfile='%s/%s' %(testpath,s)
                print(sfile)
                x=self.get_img(sfile)
                if isinstance(x,np.ndarray):
                    y=self.Predict(x)
                    y_predict=np.where(y==y.max())[1][0]
                    y_real=int(os.path.splitext(s)[0].split('_')[1]) #图片格式形如：filename_flag.png
                    #准确数统计
                    if(y_predict==y_real):
                        acc_count+=1
                    #图像个数统计
                    img_count+=1
                    #记录结果
                    f.write('%s:%d:%s\r\n'%(sfile,y_predict,y))
                    print('\tpredict:%d=>%d'%(y_real,y_predict))
                else:
                    print('\tget_img(...):=> fail!')
        #预测统计
        print('Predicts{img_count:%d,acc_count:%d,acc:%f}'%(img_count,acc_count,acc_count/img_count if img_count>0 else 0))


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
    #默认值
    default_fun=0
    default_datapath='%s/data/cifar10'%(os.getenv('HOME'))
    default_testpath='test'
    default_model='LeNet.h5'
    default_logdir='LeNet_logdir'
    default_result='LeNet_result'
    default_epochs=100
    default_dbg=0
    #程序描述
    description=''
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description)
    # Add argument
    parser.add_argument('--fun',type=int, help='脚本功能[0-训练,1-预测],默认0. eg. --fun %d'%(default_fun),default=default_fun)
    parser.add_argument('--datapath', type=str, help='数据目录. eg. --datapath "%s"'%(default_datapath), default=default_datapath)
    parser.add_argument('--testpath', type=str, help='测试目录. eg. --testpath "%s"'%(default_testpath), default=default_testpath)
    parser.add_argument('--model', type=str, help='参数文件. eg. --model "%s"'%(default_model), default=default_model)
    parser.add_argument('--log_dir', type=str, help='日志目录. eg. --log_dir "%s"'%(default_logdir), default=default_logdir)
    parser.add_argument('--result', type=str, help='预测结果. eg. --result "%s"'%(default_result), default=default_result)
    parser.add_argument('--epochs', type=int, help='迭代次数. eg. --epochs %d'%(default_epochs), default=default_epochs)
    parser.add_argument('--dbg', type=int, help='是否调试. eg. --dbg 0', default=0)
    # Parse argument
    arg = parser.parse_args()
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

