#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
基于多层感知器的softmax多分类
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

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,
          callbacks=[TensorBoard(log_dir=log_dir)])
score = model.evaluate(x_test, y_test, batch_size=128)

