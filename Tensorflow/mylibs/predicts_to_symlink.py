#!/usr/bin/env python3
import os
import numpy as np
#把预测结果转化为软链接目录结构:与具体模型，keras.processing.ImageDataGenerator配合使用
def predicts_to_symlink(preds,test_path,out_path,test_gen):
    '''把预测结果转化为软链接目录结构:与具体模型，keras.processing.ImageDataGenerator配合使用
        原始图像路径 => 图像预测结果软链接
        如：
            源图像路径 -- /src/path/class1/file1.jpg
            预测分类：class2
            创建软链接 -- /dst/path/class2/file1.jpg

    @param preds     网络预测结果,model.predict_generator返回的结果
    @param test_path 测试图像目录
    @param out_path  软链接目录
    @param test_gen  测试数据生成器
    '''
    preds_id=np.argmax(preds,axis=1) #获取预测 class_id
    class_ind_rev={v:k for k,v in test_gen.class_indices.items()} #{class_name:class_id} => {class_id:class_name}
    
    for i,pred in enumerate(preds_id):
        src='%s/%s'%(test_path,test_gen.filenames[i]) #源文件路径
        filename=os.path.basename(src)                #文件名
        classname=class_ind_rev[pred]                 #预测结果:类别名
        dst_path='%s/%s'%(out_path,classname)    
        dst='%s/%s/%s'%(out_path,classname,filename)  #软链接路径
        os.makedirs(dst_path) if not os.path.exists(dst_path) else ''
        os.symlink(src,dst)                           #创建软链接


#============================
if __name__=='__main__':
    import cv2
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import keras
    from keras import models,layers,optimizers
    from keras.preprocessing.image import ImageDataGenerator
    import keras.backend as K

    data_path='%s/work/data/gtest/classify'%os.getenv('HOME')#数据目录
    out_dir  ='%s/work/temp/fit_generator'%os.getenv('HOME')   #输出目录
    preds_dir='%s/predicts'%out_dir   #预测结果


    input_shape=(224,224,3)
    target_size=(224,224)
    epochs=1
    num_class=10
    batch_size=32

    #构造图像数据生成器:train
    gen_train = ImageDataGenerator(rescale =1./255, data_format=K.image_data_format())
    data_train=gen_train.flow_from_directory(directory='%s/train'%(data_path) ,batch_size=batch_size,target_size=target_size)
    #构造图像数据生成器:valid
    gen_valid = ImageDataGenerator(rescale =1./255, data_format=K.image_data_format())
    data_valid=gen_valid.flow_from_directory(directory='%s/valid'%(data_path) ,batch_size=batch_size,target_size=target_size)

    #构造图像数据生成器:test
    gen_test = ImageDataGenerator(rescale=1./255,   data_format=K.image_data_format())
    data_test=gen_test.flow_from_directory(directory='%s/test'%(data_path)
                                           ,batch_size=batch_size
                                           ,shuffle=False                                       
                                           ,target_size=target_size)

    
    #构建网络
    print('create model ......')
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=input_shape,name='conv2d_1'))
    model.add(layers.MaxPooling2D((2, 2),name='max_pooling2d_1'))
    model.add(layers.Flatten(name='flatten_1'))
    model.add(layers.Dense(512, activation='relu',name='dense_1'))
    model.add(layers.Dense(num_class, activation='softmax',name='dense_2'))
    #打印模型
    model.summary()
    #模型编译
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])    


    #模型训练
    print('training beginning ......')
    #模型训练
    history = model.fit_generator(
      data_train,
      steps_per_epoch=np.ceil(data_train.samples/batch_size),
      epochs=epochs,
      validation_data=data_valid,
      validation_steps=50)

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
    print('output symlinks ......')
    predicts_to_symlink(y_pred,'%s/test'%data_path,preds_dir,data_test)

    print('Done!')
