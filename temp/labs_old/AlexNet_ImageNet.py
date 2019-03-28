#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''AlexNet网络实验
    How to Check-Point Deep Learning Models in Keras
        https://machinelearningmastery.com/check-point-deep-learning-models-keras/
        https://keras-cn.readthedocs.io/en/latest/other/callbacks/#modelcheckpoint
    网络模型：AlexNet
    数 据 集：ImageNet2012
    实验结果：

    使用范例:
        #数据
            
        #训练
            $python3 AlexNet_ImageNet.py --datapath ~/data/ImageNet --trainpath train --testpath test
        #预测
            $python3 AlexNet_ImageNet.py --datapath ~/data/ImageNet --trainpath train --testpath test --fun 1
        #调试
            加参数 --dbg 1
        #Tensorboard
            日志目录:~/data/ImageNet/AlexNet_logdir
            $tensorboard --log_dir=~/data/ImageNet/AlexNet_logdir
            浏览器访问：http://localhost:6006
'''
from __future__ import print_function

import os
import sys
import pdb

import keras
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import ImageDataGenerator
from mylibs import funs
from mylibs import MyNets
import numpy as np
import argparse
import cv2


#AlexNet训练ImageNet数据集
class AlexNet_ImageNet(object):
    '''使用范例:
    obj=AlexNet_ImageNet('~/data/ImageNet','train','test','AlexNet_logdir','AlexNet.h5','AlexNet_result.txt',20)   #声明对象
    obj.LoadData()                                 #加载数据
    his=obj.Train()                                #训练
    score=obj.Evaluate()                           #评估
    obj.Predicts()                                 #预测
    obj.Report()                                   #实验报告
    '''
    #类似构造函数
    def __init__(self,data_path,train_path,valid_path,test_path,log_dir,model_file,result_file,epochs):
        '''
         @param data_path   数据目录
         @param train_path  训练目录
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
        self.train_dir=train_path                   #训练目录
        self.valid_dir=valid_path                   #校验目录
        self.test_dir=test_path                     #测试目录
        self.log_dir=log_dir                        #日志目录
        self.result_file=result_file                #预测结果文件
        self.validation_split=0.2                   #从训练数据中划分一定比例用于验证,0.0-1.0
        self.batch_size=128                         #批大小
        self.epochs=epochs                          #迭代次数
        #-------私有参数-----------
        self.model=None      #模型对象
        self.img_rows=227    #图像行数
        self.img_cols=227    #图像列数
        self.img_chs=3       #图像通道数   
        self.input_shape=(self.img_rows,self.img_cols,self.img_chs)  #输入维度
        self.train_num=60000        #训练样本数
        self.test_num=10000         #测试样本数
        self.num_classes=12         #类别数
        self.steps_per_epoch=20   #model.fit_generator参数
        self.batch_size=32          #
        
        self.train_generator=None #训练数据生成器
        self.valid_generator=None #校验数据生成器
        self.test_generator=None  #测试数据生成器
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
        #-----------------------------
        train_datagen = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            zca_epsilon=1e-4,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255,
            preprocessing_function=None,
            data_format='channels_last')
        valid_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen  = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.img_rows, self.img_cols),
                batch_size=self.batch_size,
                class_mode='categorical')
        valid_generator = valid_datagen.flow_from_directory(
                self.valid_dir,
                target_size=(self.img_rows, self.img_cols),
                batch_size=self.batch_size,
                class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(self.img_rows, self.img_cols),
                batch_size=self.batch_size,
                class_mode='categorical')

        self.train_generator=train_generator
        self.valid_generator=valid_generator
        self.test_generator =test_generator


    #创建模型
    def CreateModel(self):
        self.model=MyNets.Net_Alex(self.input_shape,self.num_classes)


    #加载模型 
    def LoadModel(self):
        if os.path.exists(self.model_file):
            model = load_model(self.model_file)
            test_datagen  = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(self.img_rows, self.img_cols),
                batch_size=self.batch_size,
                class_mode='categorical')
            self.test_generator =test_generator
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
        checkpoint_cb = ModelCheckpoint(cp_file, monitor='val_acc', verbose=1, save_best_only=True, mode='auto',period=2)
        #EarlyStopping
        earlyStopping_cb=keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0, mode='max')
        #TensorBoard
        tensorBoard_cb=TensorBoard(log_dir=self.log_dir)
        #回调函数序列
        callbacks_list = [checkpoint_cb,earlyStopping_cb,tensorBoard_cb]
        #模型训练
        self.history=self.model.fit_generator(
                self.train_generator,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.epochs,
                validation_data=self.valid_generator,
                callbacks=callbacks_list)
        #保存模型参数
        self.model.save(self.model_file)


    #模型评估
    def Evaluate(self):
        ''' 评估
        @return score:得分(loss,acc)
        '''
        self.score = self.model.evaluate_generator(self.valid_generator, steps=100, max_q_size=10, workers=1, pickle_safe=False)
        print('score{loss:%s,acc:%s}'%(self.score[0],self.score[1]))


    #-------------------------------
    def Predicts(self):
        '''模型预测,每一个文件都是一个单独的数字
           加载模型文件=>遍历预测目录，读取图形文件=>图像预测=>存放预测结果
        '''
        if self.model==None:
            print('model is none!')
            return
        if self.test_generator==None:
            print('test_generator is none!')
            return
        #测试
        vals=self.model.predict_generator(
            self.test_generator,
            steps=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            verbose=0
            )
        savefile=self.result_file           #测试结果保存文件
        img_count=0     #图像个数统计
        acc_count=0     #预测正确统计
        with open(savefile,'w') as f:
            for i, sfile in enumerate(self.test_generator.filenames):
                y=vals[i]
                y_predict=np.where(y==y.max())[0][0]
                y_real=self.test_generator.classes[i]
                #准确数统计
                if(y_predict==y_real):
                    acc_count+=1
                #图像个数统计
                img_count+=1
                #记录结果
                sline='%d,%d,[%s]:%s\r\n'%(y_real,y_predict,y,sfile)
                f.write(sline)
                print('\tpredict:%d=>%d'%(y_real,y_predict))
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
    #程序描述
    description=''
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description)
    # Add argument
    parser.add_argument('--fun',type=int, help='脚本功能[0-训练,1-预测],默认0. eg. --fun 0',default=0)
    parser.add_argument('--datapath', type=str, help='数据目录. eg. --datapath "~/data/ImageNet"', default='%s/data/ImageNet'%(os.getenv('HOME')))
    parser.add_argument('--trainpath', type=str, help='训练目录. eg. --trainpath "train"', default='train')
    parser.add_argument('--validpath', type=str, help='校验目录. eg. --validpath "valid"', default='valid')
    parser.add_argument('--testpath', type=str, help='测试目录. eg. --testpath "test"', default='test')
    parser.add_argument('--model', type=str, help='参数文件. eg. --model "AlexNet.h5"', default='AlexNet.h5')
    parser.add_argument('--log_dir', type=str, help='日志目录. eg. --log_dir "AlexNet_logdir"', default='AlexNet_logdir')
    parser.add_argument('--result', type=str, help='预测结果. eg. --result "AlexNet_result"', default='AlexNet_result')
    parser.add_argument('--epochs', type=int, help='迭代次数. eg. --epochs 20', default=100)
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
    trainpath=arg.trainpath
    validpath=arg.validpath
    testpath=arg.testpath
    model_file=arg.model
    log_dir=arg.log_dir
    result_file=arg.result
    if trainpath[0]!='/':
        trainpath='%s/%s'%(datapath,trainpath)
    if validpath[0]!='/':
        validpath='%s/%s'%(datapath,validpath)    
    if testpath[0]!='/':
        testpath='%s/%s'%(datapath,testpath)
    if model_file[0]!='/':
        model_file='%s/%s'%(datapath,model_file)
    if log_dir[0]!='/':
        log_dir='%s/%s'%(datapath,log_dir)
    if result_file[0]!='/':
        result_file='%s/%s'%(testpath,result_file)
    #----------------------------------------------
    obj=AlexNet_ImageNet(datapath,trainpath,validpath,testpath,log_dir,model_file,result_file,arg.epochs)
    if arg.fun==0:
        obj.LoadData()          #加载数据
        obj.CreateModel()       #创建模型
        obj.Train()             #训练
        score=obj.Evaluate()   #评估
        obj.Predicts()          #预测
        obj.Report()            #实验报告
    else:
        obj.LoadModel()         #加载模型
        obj.Predicts()          #预测


#==============================
if __name__=='__main__':
    arg=params() #命令行参数解析
    main(arg)

