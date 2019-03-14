#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''测试中的函数集
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
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from mylibs.ProcessBar import ShowProcess
from mylibs import funs
keras.__version__

#===============================

#数据生成器
#def DataGen(from_path,to_path,split_num,img_width=150,img_height=150,batch_size=32,enhance=False,class_mode='binary'):
def DataGen(from_path,to_path,**kws):
    '''数据生成器
    @param from_path  源目录，图像以类别名称建立子目录存储，如“cat,dog”
    @param to_path    训练目录,从from_path提取文件存放在to_path对应目录
    ----------关键字参数-----------
    @param reset      重置数据集，把to_path删除，重新分割训练
    @param split_num  训练，校验，测试分集比例(<1)或个数(>1)，如："0.6,0.2,0.2"或"100,20,20"    
    @param img_width  生成图像宽度
    @param img_height 生成图像高度
    @param enhance    是否使用数据增强
    @param class_mode 分类方法，'binary':二分类,'categorical':多分类
    使用范例：
        (train_gen,valid_gen,test_gen)=DataGen("./from","./to",reset=False,split_num="0.6,0.2,0.2",img_width=150,img_height=150,enhance=False,class_mode='binary')
        或
        (train_gen,valid_gen,test_gen)=DataGen("./from","./to",split_num="100,20,20",img_width=150,img_height=150,enhance=False)
        -------
    数据范例：
        源目录：
            ./from_path/cats/*
            ./from_path/dogs/*
        目的目录：
            ./to_path/train/cats/*
            ./to_path/train/dogs/*
            ./to_path/valid/cats/*
            ./to_path/valid/dogs/*
            ./to_path/test/cats/*
            ./to_path/test/dogs/*
    '''
    #参数初始化
    reset=False             if kws.get('reset')     ==None else kws.get('reset')
    split_num='0.6,0.2,0.2' if kws.get('split_num') ==None else kws.get('split_num')
    img_width=150           if kws.get('img_width') ==None else kws.get('img_width')
    img_height=150          if kws.get('img_height')==None else kws.get('img_height')
    batch_size=32           if kws.get('batch_size')==None else kws.get('batch_size')
    enhance=False           if kws.get('enhance')   ==None else kws.get('enhance')
    class_mode='binary'     if kws.get('class_mode')==None else kws.get('class_mode')

    for k,v in kws.items():
        print('%s:%s'%(k,v))
    #删除lab_path
    if reset:
        print('delete folder:%s'%(to_path))
        shutil.rmtree(to_path) if os.path.exists(to_path) else ''
    #图像分集
    if not os.path.exists(to_path):
        print('imgages_split:%s=>%s'%(from_path,to_path))
        funs.images_split(from_path,to_path,"train,valid,test",split_num)
    
    #数据生成器-训练数据集
    if enhance:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            '%s/train'%(to_path),
            # All images will be resized to 150x150
            target_size=(img_height, img_width),
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode=class_mode)    
    #数据生成器-校验数据集
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(
            '%s/valid'%(to_path),
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode=class_mode)    
    #数据生成器-测试数据集(测试集数据不能打乱)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            '%s/test'%(to_path),
            target_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False, #测试集数据不能打乱
            class_mode=class_mode)
    #返回数据生成器
    return (train_generator,valid_generator,test_generator)


#把数据图像化输出
def deprocess_image(x):
    '''数据图像化输出
    @param x  数据
    @return   图像化输出结果
    处理流程：
        1. x归一化处理：mean=0,std=0.1
        2. x整体抬升0.5,并作[0,1]裁剪
        3. x整体乘于255,并作[0,255]裁剪
    使用范例：
        x=np.random.rand(4,5)
        y=deprocess_image(x)
    '''
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


#生成过滤器的最大化输入
def generate_pattern(model,layer_name, filter_index,steps=40, img_width=150,img_height=150):
    '''生成过滤器的最大化输入
        通过迭代steps次梯度值，获取使网络输入最大化的输入
    @param model        网络模型
    @param layer_name   网络层名称，通过名称获取网络层
    @param filter_index 卷积核索引
    @param steps        构造迭代次数
    @param img_width    网络输入图像宽度
    @param img_height   网络输入图像高度
    @return 网络输入图像数据
    '''
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output  #网络层输出
    loss = K.mean(layer_output[:, :, :, filter_index]) #构造损失函数

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]          #梯度计算(?,150,150,3)

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  #梯度归一化处理

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads]) #迭代函数
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, img_height, img_width, 3)) * 20 + 128. #初始化网络输入图像

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(steps):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step  #为什么不是 -=
        
    img = input_img_data[0]
    return deprocess_image(img)


#图像预处理
def preprocess_input(x):
    '''图像预处理,tf格式
    圖片預處理使用Keras applications 的 preprocess_input
    https://medium.com/@sci218mike/%E5%9C%96%E7%89%87%E9%A0%90%E8%99%95%E7%90%86%E4%BD%BF%E7%94%A8keras-applications-%E7%9A%84-preprocess-input-6ef0963a483e
    
    @param x 图像数据 uint8,0-255
    '''
    return x/127.5 -1


#预测值解码
def decode_predictions(preds,files_name,class_indices_rev, top=3):
    '''预测值解码
    把model.predict的预测结果翻译为对应分类名称
    @param preds model.predict预测结果值
    @param files_name 图像文件
    @param class_indices_rev 分类字典,如{"cat":0,"dog":1}
    @return 预测解码值: {y_pred:"class_name,file_name"}
    '''
    print(preds)
    print(preds[:])
    print(class_indices_rev)
    print(files_name)
    if preds.shape[-1]==1:
        preds_fix=[0 if x<0.5 else 1 for x in preds]
    else:
        preds_fix=[np.argmax(x)  for x in preds]
    print(preds_fix)
    #y={x:'%s,%s'%(class_indices_rev[0] if x <0.5 else class_indices_rev[1],files_name[n]) for n, x in enumerate(preds[:])}
    y={x:'%s,%s'%(class_indices_rev[x],files_name[n]) for n, x in enumerate(preds_fix)}
    return y


#可视化训练曲线
def visualizer_scalar(history):
    '''绘制训练曲线 epochs-acc,epochs-loss
    @param history  model.fit 训练日志
    '''
    #训练曲线
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


#可视化FeatureMap
def visualizer_feature_map(model,test_img_path,target_size=(150,150),images_per_row=16,img_margin=3):
    '''可视化FeatureMap
    @param model          网络模型
    @param teste_img_path 测试图像路径
    @param target_size    图像修正尺寸
    @param images_per_row 网格图像每行显示图像单元个数
    @param img_margin     图像单元间隙
    '''
    #加载测试图像
    img = image.load_img(test_img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    # Its shape is (1, 150, 150, 3)
    #显示原始测试图像
    plt.figure()
    plt.imshow(img_tensor[0])
    plt.show()

    #自动寻找最后一个卷积层
    layer_outputs=[]
    top_num=0
    for n,layer in enumerate(model.layers):
        if len(layer.output.shape)==4:
            layer_outputs.append(layer)
        else:
            top_num=n
            break;
    #FeatureMap网络层
    layer_outputs = [layer.output for layer in model.layers[:top_num]]         #前面N个输出层
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) #前面N层输出模型
    activations = activation_model.predict(img_tensor)                         #前面N层模型预测
    layer_names=[layer.name for layer in model.layers[:top_num]]               #前面N层名称
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1] #卷积核个数

        # The feature map has shape (1, size, size, n_features)
        size_width = layer_activation.shape[2]        #FeatureMap大小
        size_height = layer_activation.shape[1]        #FeatureMap大小

        # We will tile the activation channels in this matrix
        rows = n_features // images_per_row   #图像单元行数
        cols=images_per_row                   #图像单元列数

        #初始化图像网格[rows{height} x cols{width}]
        display_grid = np.zeros((rows*size_height+(rows-1)*img_margin, cols * size_width+(cols-1)*img_margin))  

        # We'll tile each filter into this big horizontal grid
        #把FeatureMap按顺序显示在大图网格，每行images_per_row个FeatureMap
        for row in range(rows):
            for col in range(cols):
                #提取FeatureMap
                channel_image = layer_activation[0,
                                                 :, :,
                                                 row * cols + col]
                # Post-process the feature to make it visually palatable
                #FeatureMap显示优化处理
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                #把FeatureMap拷贝到相应图像单元区间
                row_start=row*size_height+row*img_margin
                row_end=row_start+size_height
                col_start=col*size_width+col*img_margin
                col_end=col_start+size_width
                #print(display_grid.shape)
                #print('row_start:%d,row_end:%d,col_start:%d,col_end:%d'%(row_start,row_end,col_start,col_end))
                display_grid[row_start : row_end,col_start:col_end] = channel_image

        # Display the grid
        plt.figure(figsize=(1./size_width  * display_grid.shape[1], #width
                            1./size_height * display_grid.shape[0]))#height
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')    

    plt.show()

    
#可视化过滤器最大化网络输入
def visualizer_filter_input(model,layers_name,gen_pat_steps=40,images_per_row=16,img_width=150,img_height=150,img_margin=3):
    '''可视化过滤器最大化的网络输入
    @param model          网络模型
    @param gen_pat_steps  梯度计算迭代次数
    @param images_per_row 网格图像每行显示图像单元个数
    @param img_width      输入图像宽度
    @param img_height     输入图像高度
    @param img_margin     网格图像单元间隙
    @param layers_name    卷积层向量表
    '''
    for layer_name in layers_name:
        #type(n_features)#=> <class 'tensorflow.python.framework.tensor_shape.Dimension'>
        n_features=model.get_layer(layer_name).output.shape[-1] #卷积核个数
        rows = n_features // images_per_row   #图像单元行数
        rows=rows.value
        cols=images_per_row                   #图像单元列数

        #初始化图像网格[rows{height} x cols{width}]
        display_grid = np.zeros((rows*img_height+(rows-1)*img_margin, cols * img_width+(cols-1)*img_margin,3))  

        # We'll tile each filter into this big horizontal grid
        print('Generating layer of %s ......'%(layer_name))
        pbar=ShowProcess(100)
        for row in range(rows):
            for col in range(cols):
                filter_index=row*cols+col
                filter_img = generate_pattern(model,layer_name, filter_index,steps=gen_pat_steps, img_width=img_width,img_height=img_height)
                row_start=row*img_height+row*img_margin
                row_end=row_start+img_height
                col_start=col*img_width+col*img_margin
                col_end=col_start+img_width
                #print(display_grid.shape)
                #print('row_start:%d,row_end:%d,col_start:%d,col_end:%d'%(row_start,row_end,col_start,col_end))            
                display_grid[row_start : row_end,col_start:col_end] = filter_img
                pbar.show_process((filter_index*100/(rows*cols)))
        pbar.show_process(100)
        # Display the results grid
        plt.figure(figsize=(20, 20))
        plt.imshow(display_grid)
        plt.show()

        
#可视化类激活热力图
def visualizer_heatmap(model,test_img_path,last_conv_layer_name,target_size=(150,150)):
    '''可视化类激活热力图
    @param model                网络模型
    @param test_img_path        测试图像路径
    @param last_conv_layer_name 最后一个卷积层
    @param target_size          图像修正尺寸大小
    '''
    #加载测试图像
    img = image.load_img(test_img_path, target_size=target_size) #加载测试图像,PIL格式
    x = image.img_to_array(img)                                              #转为numpy格式
    x = np.expand_dims(x, axis=0) #增加samples维,构成batch为1的样本集 (1,150,150,3)
    x = preprocess_input(x)       #图像数组预处理
    preds = model.predict(x)      #模型预测
    #=======================
    pred_index=np.argmax(preds[0])                             #获取预测结果的标签索引，对于二分类，返回0
    weight_output = model.output[:, pred_index]                #预测结果对应的权重向量(?,)
    last_conv_layer = model.get_layer(last_conv_layer_name)    #最后一个卷积层,last_conv_layer.output=>[(?,15,15,128)]
    grads = K.gradients(weight_output, last_conv_layer.output)[0] #计算梯度值(?,15,15,128)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))               #每个通道取梯度平均值(128,)
    #weight_output=>预测神经元的向量权重
    #last_conv_layer.output[0]=>最后一个卷积层输出的FeatureMap
    #grads=>weight_output对last_conv_layer.output的梯度计算
    #pooled_grads=>每个过滤器的梯度均值
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]]) #构造迭代函数
    pooled_grads_value, conv_layer_output_value = iterate([x])    #执行迭代函数

    #FeatureMap中每个通道乘于通道的重要程度
    for i in range(last_conv_layer.output.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    #通道平均值即为热力图
    heatmap = np.mean(conv_layer_output_value, axis=-1)    
    #================
    #热力图可视化
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    #==================
    #热力图-原始图融合可视化
    import cv2
    # We use cv2 to load the original image
    img = cv2.imread(test_img_path)
    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img
    # Save the image to disk
    #cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)
    img_hot_path='%s_hot%s'%(os.path.splitext(test_img_path)[0],os.path.splitext(test_img_path)[1])
    cv2.imwrite(img_hot_path, superimposed_img)

    img = image.load_img(img_hot_path, target_size=target_size)
    plt.figure()
    plt.imshow(img)
    plt.show()     

#准确率计算
def preds_acc(preds,data_gen):
    '''准确率计算
    @param preds   model.predict预测结果
    @param data_gen 数据生成器
    @return 准确率
    '''
    ##测试准确率
    #y_pred=y_pred.reshape(y_pred.size,1)
    #classes=data_gen.classes.reshape(data_gen.classes.size,1).astype('float32')
    #y_pred[y_pred<0.5]=0
    #y_pred[y_pred>=0.5]=1
    #y_sub_abs=np.abs(classes-y_pred)
    #y_acc=1.0-y_sub_abs.sum()/y_sub_abs.size
    #return y_acc

    #测试准确率
    if preds.shape[-1]==1:
        #二分类
        preds_index=[0 if x<0.5 else 1 for x in preds]
    else:
        #多分类
        preds_index=[np.argmax(x) for x in preds]
        
    preds_sub=preds_index-data_gen.classes
    acc=1.0-np.count_nonzero(preds_sub)/preds_sub.size
    return acc
    
    
#网络测试结果
def preds_result(preds,data_gen):
    '''网络测试结果
    @param preds    model.predict返回结果
    @param data_gen 数据生成器
    @return {pred_index:"pred_vec,class_name"}
    '''
    classes_inds_rev={v:k for k,v in data_gen.class_indices.items()} #把{name:index}反转{index:name}
    if preds.shape[-1]==1:
        #二分类
        rets={data_gen.filenames[n]:'%s-%s'%(x,classes_inds_rev[0 if x<0.5 else 1]) for n,x in enumerate(preds)}
    else:
        #多分类
        rets={data_gen.filenames[n]:'%s-%s'%(x,classes_inds_rev[np.argmax(x)]) for n,x in enumerate(preds)}
    return rets
        

#图像分类预测
def predict_images(model,images,class_indices,img_width=150,img_height=150,show_img=True):
    preds_val={}
    class_indices_rev={v:k for k,v in class_indices.items()}
    for test_img_path in images:
        #加载测试图像
        img = image.load_img(test_img_path, target_size=(img_height, img_width)) #加载测试图像,PIL格式
        x = image.img_to_array(img)                                              #转为numpy格式
        x = np.expand_dims(x, axis=0) #增加samples维,构成batch为1的样本集 (1,150,150,3)
        x = preprocess_input(x)       #图像数组预处理
        preds = model.predict(x)      #模型预测
        if show_img:
            print('Predicted:', decode_predictions(preds,[test_img_path],class_indices_rev, top=3)) #预测结果
            #显示原始图像
            plt.figure()
            plt.imshow(img)
            plt.show()
        else:
            if preds.shape[-1]==1:
                preds_val[test_img_path]=class_indices_rev[0 if preds[0]<0.5 else 1]
            else:
                preds_val[test_img_path]=class_indices_rev[np.argmax(preds[0])]
    return preds_val
        
#保存训练结果
def save_model(model,to_path,class_indices):
    '''保存训练结果
    @param model         网络模型
    @param train_gen     训练数据生成器  
    @param to_path       保存目录
    @param class_indices 训练类别
    
    训练模型保存文件路径
    to_path/model.h5
    训练数据生成器保存文件路径
    to_path/config.json
    '''
    model.save('%s/model.h5'%(to_path))
    json.dump(class_indices,open('%s/config.json'%(to_path),'w'))
    

