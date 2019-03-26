#!/usr/bin/env python3
# coding: utf-8

# # InceptionV3模型训练
#    
# ### 实验内容
#     InceptionV3模型迁移与fine-tune,训练新数据集
#     
# ### 实验目的
#     InceptionV3模型应用
#     
# ### 实验步骤
#     1.构造数据生成器
#         gen_train =ImageDataGenetrator(...)
#         data_train=gen_train.flow_from_directory(...)
#         valid_gen =ImageDataGenetrator(...)
#         data_valid=gen_valid.flow_from_directory(...)
#         test_gen  =ImageDataGenetrator(...)
#         data_test =gen_test.flow_from_directory(...)
#     2.加载 tf.keras.applications.inception_v3 ,include_top=False
#         base_model = InceptionV3(weights='imagenet',include_top=False)
#     3.增加新的输出层
#         x = base_model.output
#         x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
#         x = Dense(1024,activation='relu')(x)
#         predictions = Dense(10,activation='softmax')(x)
#         model = Model(inputs=base_model.input,outputs=predictions)    
#     4.训练模型
#       4.1. 迁移模型
#       4.2. fine-tune
#     5.测试模型
#     
# ### 实验数据
#     抽取ImageNet子集10子分类作为实验数据
#     
#     
# ### 实验结果
# 
# ### 参考资料
# 

# # GPU配置

# In[1]:


#禁用GPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #-1:禁用,0-n启用第几块显卡，多个以逗号隔开

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

#ConfigProto配置
config = tf.ConfigProto()

#设置GPU的百分比，程序需要还是会突破阈值
#config.gpu_options.per_process_gpu_memory_fraction = 1 #0-1之间的浮点数表示占用百分比

#GPU按需使用,不全部占满显存, 按需分配
#config.gpu_options.allow_growth=True #True:按需分配,False:一次性满额分配

# 设置session
sess = tf.Session(config=config)
KTF.set_session(sess)


# # 参数配置

# In[2]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from   keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from   keras import models,layers,optimizers
from   mylibs.predicts_to_symlink import predicts_to_symlink

retrain=1                #是否重新训练模型
gap_layer=0              #fine-tune起始层
dataset='imagenet/full'  #数据集
include_top=True         #是否包含头部链接
#数据目录
data_path='%s/work/data/%s'%(os.getenv('HOME'),dataset)
#输出目录
out_path='%s/work/data/labs_out/lab_inceptionV3/%s-%d-%d'%(os.getenv('HOME'),dataset.replace('/','-'),retrain,gap_layer)

log_path  ='%s/log_dir'%out_path    #输出日志目录
preds_path='%s/predicts'%out_path   #预测结果
cp_file='%s/cp_file.h5'%out_path    #训练断点
model_file='%s/model.h5'%out_path   #模型文件

os.makedirs(out_path)   if not os.path.exists(out_path)   else ''
os.makedirs(log_path)   if not os.path.exists(log_path)   else ''
os.makedirs(preds_path) if not os.path.exists(preds_path) else ''

input_shape=(299,299,3)
target_size=(299,299)
epochs=100
num_class=1000
batch_size=32


# # 构造数据生成器

# In[3]:


from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

#构造图像数据生成器:train
gen_train = ImageDataGenerator(
        featurewise_center           =True,
        samplewise_center            =False,
        featurewise_std_normalization=True,
        samplewise_std_normalization =False,
        zca_whitening                =False,
        zca_epsilon                  =1e-6,
        rotation_range               =30,
        width_shift_range            =0.3,
        height_shift_range           =0.3,
        shear_range                  =0.3,
        zoom_range                   =0.3,
        channel_shift_range          =0.,
        fill_mode                    ='nearest',
        cval                         =0.,
        horizontal_flip              =True,
        vertical_flip                =True,
        rescale                      =1./255,
        preprocessing_function       =None,
        data_format                  =K.image_data_format()
       )
data_train=gen_train.flow_from_directory(directory='%s/train'%(data_path)
                                         ,batch_size=batch_size
                                         ,target_size=target_size)
#构造图像数据生成器:valid
gen_valid = ImageDataGenerator(
        featurewise_center           =True,
        samplewise_center            =False,
        featurewise_std_normalization=True,
        samplewise_std_normalization =False,
        zca_whitening                =False,
        zca_epsilon                  =1e-6,
        rotation_range               =30,
        width_shift_range            =0.3,
        height_shift_range           =0.3,
        shear_range                  =0.3,
        zoom_range                   =0.3,
        channel_shift_range          =0.,
        fill_mode                    ='nearest',
        cval                         =0.,
        horizontal_flip              =True,
        vertical_flip                =True,
        rescale                      =1./255,
        preprocessing_function       =None,
        data_format                  =K.image_data_format()
       )
data_valid=gen_valid.flow_from_directory(directory='%s/valid'%(data_path)
                                         ,batch_size=batch_size
                                         ,target_size=target_size)

#构造图像数据生成器:test
gen_test = ImageDataGenerator(
        featurewise_center           =False,
        samplewise_center            =False,
        featurewise_std_normalization=False,
        samplewise_std_normalization =False,
        zca_whitening                =False,
        zca_epsilon                  =1e-6,
        rotation_range               =0.,
        width_shift_range            =0.,
        height_shift_range           =0.,
        shear_range                  =0.,
        zoom_range                   =0.,
        channel_shift_range          =0.,
        fill_mode                    ='nearest',
        cval                         =0.,
        horizontal_flip              =False,
        vertical_flip                =False,
        rescale                      =1./255,
        preprocessing_function       =None,
        data_format                  =K.image_data_format()
       )
data_test=gen_test.flow_from_directory(directory='%s/test'%(data_path)
                                       ,batch_size=batch_size
                                       ,shuffle=False                                       
                                       ,target_size=target_size)


# # 构建模型

# In[4]:


from keras import models,layers,optimizers
from keras.applications.inception_v3 import InceptionV3

# 构建基础模型
if retrain:
    #重新训练
    base_model = InceptionV3(weights=None,include_top=include_top)
else:
    #调参模式
    base_model = InceptionV3(weights='imagenet',include_top=include_top)

if include_top:
    model=base_model
else:    
    # 增加新的输出层
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
    x = layers.Dense(1024,activation='relu')(x)
    predictions = layers.Dense(num_class,activation='softmax')(x)
    model = models.Model(inputs=base_model.input,outputs=predictions)

#打印模型
model.summary()
for i,layer in enumerate(model.layers):
    print('%d:%s-%s'%(i,layer.name,layer.trainable))


# # 训练模型

# In[5]:


#迁移学习，适应新数据集
def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#fine-tune,调整高层参数
def setup_to_fine_tune(model,base_model,GAP_LAYER=17):
    #GAP_LAYER = 17 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=optimizers.Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
#加载权重    
def load_weights(model,base_model,model_file):
    for layer in base_model.layers:
        layer.trainable = True
    if os.path.exists(model_file):
        print('load weights:',model_file)
        model.load_weights(model_file,by_name=True)
#------------------------
#加载权重参数
load_weights(model,base_model,cp_file)
#------------------------
#回调函数序列
#断点训练:monitor监控参数可以通过score = model.evaluate(x_test, y_test, verbose=0)的score查询
checkpoint_cb = keras.callbacks.ModelCheckpoint(cp_file, monitor='val_acc', verbose=1, save_best_only=True, mode='auto',period=1)
#EarlyStopping
#earlyStopping_cb=keras.callbacks.EarlyStopping(monitor='acc', patience=100, verbose=0, mode='max')
#TensorBoard
tensorBoard_cb=keras.callbacks.TensorBoard(log_dir=log_path)
#回调函数序列
#callbacks_list = [checkpoint_cb,earlyStopping_cb,tensorBoard_cb]
callbacks_list = [checkpoint_cb,tensorBoard_cb]
#-----------------------
#迁移学习
if gap_layer>0:
    print('transfer learning ......')
    setup_to_transfer_learning(model,base_model)
    history_tl = model.fit_generator(generator=data_train,
                        steps_per_epoch=data_train.samples//batch_size,
                        epochs=5,#2
                        validation_data=data_valid,
                        validation_steps=12,#12
                        class_weight='auto',
                        callbacks=callbacks_list
                        )
#-----------------------    
#fine-tune
print('fine tuning ......')
setup_to_fine_tune(model,base_model,gap_layer)
history_ft = model.fit_generator(generator=data_train,
                                 steps_per_epoch=data_train.samples//batch_size,
                                 epochs=1000,
                                 validation_data=data_valid,
                                 validation_steps=1,
                                 class_weight='auto',
                                 callbacks=callbacks_list)


# In[6]:


#-----------------26 11-----
#保存模型
print('save model:',model_file)
model.save(model_file)


# # 测试模型

# In[7]:


import shutil
from mylibs.predicts_to_symlink import predicts_to_symlink
#计算精度
def compute_acc(y_pred,y_true):
    acc=(y_pred-y_true)==0
    return acc.sum()/acc.size

#加载模型
model.load_weights(model_file)

#模型测试
print('predicting beginning ......')
#type(y_pred)=> <class 'numpy.ndarray'>
y_pred=model.predict_generator(
    data_test, 
    steps=None, #预测轮数
    max_queue_size=32, 
    workers=1, 
    use_multiprocessing=False, 
    verbose=1)

#输出软链接目录
shutil.rmtree(preds_path) if os.path.exists(preds_path) else ''
predicts_to_symlink(y_pred,'%s/test'%data_path,preds_path,data_test)

#准确率计算
acc=compute_acc(np.argmax(y_pred,axis=1),data_test.classes)
print('samples:',data_test.samples)
print('classes[:2]:')
print(data_test.classes[:2])
print('y_pred.shape:',y_pred.shape)
print('y_pred[:2]:')
print(y_pred[:2])
print('准确率:',acc)


# In[ ]:




