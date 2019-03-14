#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''利用ResNet50网络进行ImageNet分类

'''

import os
import sys
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from mylibs import funs
from mylibs import ProcessBar

def Predicts(path):
    #收集图像文件
    files=funs.GatherFiles(path,exts='.jpg,.jpeg,.png')
    pb=ProcessBar.ShowProcess(100)
    model = ResNet50(weights='imagenet')
    nFiles=len(files)
    for i,sfile in enumerate(files):
        img_path = sfile
        img = image.load_img(img_path, target_size=(224, 224))
        if img==None:
            print('img.load_img(%s)=None'%(img_path))
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        print(sfile)
        print('Predicted:', decode_predictions(preds, top=3)[0])
        # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
        pb.show_process(int(i*100/nFiles))
    pb.show_process(100)

if __name__=='__main__':
    if len(sys.argv)<2:
        print('usge:%s path'%(sys.argv[0]))
    else:
        Predicts(sys.argv[1])

