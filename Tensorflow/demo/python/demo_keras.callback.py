#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''自定义回调函数范例
  https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
'''

import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import numpy as np

#数据目录
log_dir='%s/data/demo/%s/log'%(os.getenv('HOME'),sys.argv[0].split('.')[0])
os.makedirs(log_dir) if not os.path.exists(log_dir) else None

#回调函数范例
class demo_callback(keras.callbacks.Callback):
    def __init__(self,arg1=None,arg2=None):
        '''构造函数
        self: ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', 
               '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__ne__', 
               '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', 
               '__subclasshook__', '__weakref__', 'arg1', 'arg2', 'model', 'on_batch_begin', 'on_batch_end', 
               'on_epoch_begin', 'on_epoch_end', 'on_train_begin', 'on_train_end', 'set_model', 'set_params', 'validation_data']
        self.model: ['__bool__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', 
               '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', 
               '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']

        '''
        super(demo_callback,self).__init__() #调用父类构造函数
        self.arg1=arg1  #回调函数传入的参数
        self.arg2=arg2  #回调函数传入的参数

        print('demo_callback.arg1:%s,arg2:%s'%(arg1,arg2))
        print('self:',dir(self))
        print('self.model:',dir(self.model))

    #on_epoch_begin: 在每个epoch开始时调用
    def on_epoch_begin(self,epoch,logs=None):
        '''
        self: ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', 
               '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__ne__', 
               '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', 
               '__subclasshook__', '__weakref__', 'arg1', 'arg2', 'model', 'on_batch_begin', 'on_batch_end', 
               'on_epoch_begin', 'on_epoch_end', 'on_train_begin', 'on_train_end', 'params', 'set_model', 'set_params', 'validation_data']
        epoch: 0
        logs: {}
        self.params: ['__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', 
                      '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', 
                      '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
                      '__repr__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 
                      'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
        samples:1000
        epochs:4
        do_validation:False
        metrics:['loss', 'acc']
        steps:None
        verbose:1
        batch_size:32
        '''
        print('demo_callback.on_epoch_begin')
        print('self:',dir(self))
        print('epoch:',epoch)
        print('logs:',logs)
        print('self.params:',dir(self.params))
        for k,v in self.params.items():
            print('%s:%s'%(k,v))

    #on_epoch_end: 在每个epoch结束时调用
    def on_epoch_end(self,epoch,logs=None):
        '''
        epoch: 3
        logs: {'acc': 0.108, 'loss': 2.304810094833374}
        demo_callback.on_train_end
        100/100 [==============================] - 0s 458us/step

        '''
        print('demo_callback.on_epoch_end')
        print('epoch:',epoch)
        print('logs:',logs)

    #on_batch_begin: 在每个batch开始时调用
    def on_batch_begin(self,batch,logs=None):
        '''
        batch: 31
        logs: {'batch': 31, 'size': 8}
        self.params: ['__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
                      '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__iter__', '__le__', '__len__', '__lt__', 
                      '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setitem__', '__sizeof__', '__str__', 
                      '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
        samples:1000
        epochs:4
        do_validation:False
        metrics:['loss', 'acc']
        steps:None
        verbose:1
        batch_size:32
        '''
        print('demo_callback.on_batch_begin')
        print('batch:',batch)
        print('logs:',logs)
        print('self.params:',dir(self.params))
        for k,v in self.params.items():
            print('%s:%s'%(k,v))

    #on_batch_end: 在每个batch结束时调用
    def on_batch_end(self,batch,logs=None):
        '''
        batch: 31
        logs: {'acc': 0.0, 'batch': 31, 'loss': 2.321695, 'size': 8}
        self.params: ['__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', 
                      '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', 
                      '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
                      '__repr__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 
                      'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
        samples:1000
        epochs:4
        do_validation:False
        metrics:['loss', 'acc']
        steps:None
        verbose:1
        batch_size:32
        1000/1000 [==============================] - 0s 151us/step - loss: 2.3048 - acc: 0.1080
        '''
        print('demo_callback.on_batch_end')
        print('batch:',batch)
        print('logs:',logs)
        print('self.params:',dir(self.params))
        for k,v in self.params.items():
            print('%s:%s'%(k,v))

    #on_train_begin: 在训练开始时调用
    def on_train_begin(self,logs=None):
        print('demo_callback.on_train_begin')

    #on_train_end: 在训练结束时调用
    def on_train_end(self,logs=None):
        print('demo_callback.on_train_end')


#构造测试数据
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

#创建网络
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(20,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#优化方法
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#回调函数
tensorboard_cb=TensorBoard(log_dir=log_dir)
demo_cb=demo_callback('arg1','arg2')
cbs=[tensorboard_cb,demo_cb]

#训练模型
model.fit(x_train, y_train,
          epochs=4,
          batch_size=32,
          callbacks=cbs)

#模型评估
score = model.evaluate(x_test, y_test, batch_size=32)

