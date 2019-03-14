#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''批量训练网络
'''

import keras
import os
import json
import shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from mylibs.ProcessBar import ShowProcess
from mylibs import funs
from mylibs.my_contrib import *
keras.__version__

##实验参数
print('\n==============================================')
print('设置实验参数')
lab_name='猫狗分类实验-VGG16预训练模型调参'              #实验名称
data_path='%s/data/cats_and_dogs'%(os.getenv('HOME')) #猫狗分类数据根目录
ori_path='%s/ori'%(data_path)                         #猫狗分类原始文件目录
lab_path='%s/lab_vgg19_fine_tuning'%(data_path)          #实验方案目录
split_num="10000,2000,2000"                           #实验数据分割方案,<1：比例分割，>1：数量分割
batch_size=32                                         #批量大小
data_enhance=False                                    #ImageDataGenerator数据启用数据增强
epochs=10                                             #训练轮次
img_width=224                                         #训练图像宽度
img_height=224                                        #训练图像高度 
test_img_path='%s/test.jpg'%(data_path)               #测试图片路径
images_per_row=16       #图像显示每行显示的单元个数
#feature_map_top_num=12  #FeatureMap前面N层{include_top=False}
img_margin=3            #图像单元空隙
layers_name=['conv2d_1','conv2d_2','conv2d_3','conv2d_4'] #卷积层名称
#layers_name=['conv2d_1'] #卷积层名称
last_conv_layer_name='conv2d_4' #最后一层卷积层
gen_pat_steps=40                           #构造迭代次数
cp_file='%s/checkpoint.h5'%(lab_path)      #ModelCheckpoint 文件路径
his_file='%s/history.json'%(lab_path)      #训练日志文件路径
class_mode='binary'                        #分类方法,'binary':二分类，'categorical':多分类
loss='binary_crossentropy'  #损失函数,'binary_crossentropy':二分类，'categorical_crossentropy':多分类

test_cat_path='%s/test_cat.jpg'%(data_path) #猫的测试图像
test_dog_path='%s/test_dog.jpg'%(data_path) #狗的测试图像

##加载数据
print('\n==============================================')
print('加载数据......')
#删除lab_path
#shutil.rmtree(lab_path) if os.path.exists(lab_path) else ''

#数据生成器
(train_gen,valid_gen,test_gen)=DataGen(ori_path,lab_path,reset=True,split_num=split_num
                                   ,img_width=img_width,img_height=img_height
                                   ,batch_size=batch_size,enhance=data_enhance,class_mode=class_mode)


print('\n==============================================')
print('构建网络')
from keras.applications import ResNet50
## 创建网络
conv_base=ResNet50(weights='imagenet',include_top=False,input_shape=(224, 224, 3))

'''
for layer in conv_base.layers:
    layer.trainable=False
'''

'''
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
'''

conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
#打印模型
model.summary()
#模型编译
model.compile(loss=loss,
          optimizer=optimizers.RMSprop(lr=1e-4),
          metrics=['acc'])

class train_callback(keras.callbacks.Callback):
    def __init__(self,log_file,history={},verbose=0):
        super(train_callback,self).__init__() #调用父类构造函数
        self.log_file=log_file #训练日志文件路径
        self.history=history   #训练日志
        self.verbose=verbose   #是否显示保存信息
        
    #on_epoch_end: 在每个epoch结束时调用
    def on_epoch_end(self,epoch,logs=None):
        #最佳日志
        if len(self.history)==0:
            for k,v in logs.items():
                self.history[k]=[v]
        else:
            for k,v in logs.items():
                self.history[k].append(v)
        #保存日志
        json.dump(self.history,open(self.log_file,'w'))
        if self.verbose==1:
            print('更新训练日志(%d):%s'%(len(self.history),self.log_file))

##网络训练
print('\n==============================================')
print('网络训练 ......')
#加载断点
if os.path.exists(cp_file):
    model.load_weights(cp_file)
    print('加载模型文件:',cp_file)
#训练日志
history2={}
if os.path.exists(his_file):
    history2=json.load(open(his_file,'r'))
    print('加载训练日志:',his_file)


#回调函数保存训练日志    
his_cb=train_callback(his_file,history=history2)

#断点训练:monitor监控参数可以通过self.score = self.model.evaluate(self.x_test, self.y_test, verbose=0)的score查询
checkpoint_cb = ModelCheckpoint(cp_file, monitor='val_acc', verbose=1, save_best_only=True, mode='auto',period=2)
#EarlyStopping
earlyStopping_cb=keras.callbacks.EarlyStopping(monitor='acc', patience=3, verbose=0, mode='max')
#TensorBoard
#tensorBoard_cb=TensorBoard(log_dir=self.log_dir)
#回调函数序列
callbacks_list = [checkpoint_cb,earlyStopping_cb,his_cb]

history = model.fit_generator(
  train_gen,
  steps_per_epoch=np.ceil(train_gen.samples/batch_size),
  epochs=epochs,
  validation_data=valid_gen,
  validation_steps=50,
  callbacks=callbacks_list)


