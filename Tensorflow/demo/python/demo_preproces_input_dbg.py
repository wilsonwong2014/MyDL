#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''v166.preprocess_input调试代码追踪
圖片預處理使用Keras applications 的 preprocess_input
    https://medium.com/@sci218mike/%E5%9C%96%E7%89%87%E9%A0%90%E8%99%95%E7%90%86%E4%BD%BF%E7%94%A8keras-applications-%E7%9A%84-preprocess-input-6ef0963a483e
    
目前为止Keras提供的pre-train model有：
    Xception,VGG16,VGG19,ResNet50,InceptionV3,InceptionResNetV2,MobileNet,DenseNet,NASNet,MobileNetV2都可以使用preprocess_input
(按照Keras doucumentation排序)
                        Input size          Data format                     mode
    -------------------------------------------------------------------------------
    Xception            299x299             channels_last                   tf
    VGG16               224x224             channels_first/channels_last    caffe
    VGG19               224x224             channels_first/channels_last    caffe
    ResNet50            224x224             channels_first/channels_last    caffe
    InceptionV3         299x299             channels_first/channels_last    tf
    InceptionResNetV2   299x299             channels_first/channels_last    tf
    MobileNet           224x224             channels_last                   tf
    DenseNet            224x224             channels_first/channels_last    torch
    NASNet              331x331/224x224     channels_first/channels_last    tf
    MobileNetV2         224x224             channels_last                   tf

使用 preprocess_input時輸入為皆為RGB values within [0, 255]
圖片預處理方式有三種caffe、tf、torch:
    caffe   : VGG16、VGG19、ResNet50
    tf      : Xception、InceptionV3、InceptionResNetV2、MobileNet、NASNet、MobileNetV2
    torch   : DenseNet

    mode = caffe 
        (will convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset)
        減去ImageNet平均 BGR [103.939, 116.779, 123.68]
    mode = tf 
        ( will scale pixels between -1 and 1 )
        除以127.5，然後減 1。
    mode = torch 
        ( will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset)
        除以255，減去ImageNet平均[0.485, 0.456, 0.406] ，除以標準差[0.229, 0.224, 0.225]。
參考資料
https://keras.io/applications/
https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/keras/applications/imagenet_utils.py
https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
    
'''

from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input, decode_predictions 
import numpy as np 

import pdb
pdb.set_trace()

x=np.random.randint(0,256,[4,5,3])
print('x-mean:%s'%(x.mean()))
print('x-std:%s'%(x.std()))

y1=preprocess_input(x)
print('y1-mean:%s'%(y1.mean()))
print('y1-std:%s'%(y1.std()))

y2=x/127.5-1
print('y2-mean:%s'%(y2.mean()))
print('y2-std:%s'%(y2.std()))

