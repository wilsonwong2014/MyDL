#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
'''模型可视化函数集
'''

import keras
from keras import layers
from keras import models
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from mylibs.ProcessBar import ShowProcess
from mylibs import funs

#绘制FeatureMap
def plot_FeatureMap(model,layers_name,img_file,target_size=(150,150),images_per_row=16,img_margin=1):
    '''绘制图像文件的FeatureMap
        model为已经训练好的模型对象
    @param model                训练好的网络模型
    @param layers_name    [list]输出层名称，输出层一般为卷积层
    @param img_file       [str ]图像文件
    @param images_per_row [int ]每行显示多少个特征图
    @param img_margin     [int ]图像单元间隙
    
    使用范例：
        model=...                                #创建/加载模型/模型编译/模型训练 
        layers_name=['conv_1','conv_2','conv_3'] #设置输出层        
        img_file='./test/1.jpg'                  #图像路径
        plot_FeatureMap(model,layers_name,img_file)
    '''
    #加载图像
    img = image.load_img(img_file, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    # Its shape is (1, 150, 150, 3)
    print(img_tensor.shape)
    plt.imshow(img_tensor[0])
    plt.title('plot_FeatureMap:source image')  #原始图
    plt.show()

    #创建输出模型
    layer_outputs = [model.get_layer(sname).output for sname in layers_name] #输出层
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) #前面N层输出模型
    activations = activation_model.predict(img_tensor)                         #前面N层模型预测    
    for layer_name, layer_activation in zip(layers_name, activations):        
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1] #卷积核个数

        # The feature map has shape (1, size, size, n_features)
        size_width = layer_activation.shape[-2]        #FeatureMap大小
        size_height = layer_activation.shape[-3]        #FeatureMap大小

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
                print('layer_activation.shape:',layer_activation.shape)
                print('channel_image.shape:',channel_image.shape)
                print('row_start:%d,row_end:%d,col_start:%d,col_end:%d'%(row_start,row_end,col_start,col_end))
                display_grid[row_start : row_end,col_start:col_end] = channel_image

        # Display the grid
        plt.figure(figsize=(1./size_width  * display_grid.shape[1], #width
                            1./size_height * display_grid.shape[0]))#height
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')    
        plt.title('plot_FeatureMap(layer_name:%s)'%(layer_name))
    plt.show() 
    
#可视化卷积神经网络的过滤器
def plot_layer_filter(model,layers_name,target_size=(150,150),img_margin=1,images_per_row=16,gen_pat_steps=20):
    '''可视化卷积神经网络的过滤器
    @param model                训练好的网络模型
    @param layers_name    [list]输出层名称,一般为卷积层
    @param img_width      [int ]过滤器图像宽度
    @param img_height     [int ]过滤器图像高度
    @param img_margin     [int ]图像单元间隙
    @param images_per_row [int ]每行显示图像单元个数
    @param gen_pat_steps  [int ]生成过滤器图像迭代次数
    
    使用范例：
        model=...                                #创建/加载模型/模型编译/模型训练 
        layers_name=['conv_1','conv_2','conv_3'] #设置输出层
        plot_layer_filter(model,layers_name)
        
    '''
    #把数据图像化输出
    def deprocess_image(x):
        '''数据图像化输出
        @param x  数据
        @return   图像化输出结果
        处理流程：
            1. x归一化处理：mean=0,std=0.1
            2. x整体抬升0.5,并作[0,1]裁剪
            3. x整体乘于255,并作[0,255]裁剪
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
        grads = K.gradients(loss, model.input)[0]          #梯度计算

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
            input_img_data += grads_value * step

        img = input_img_data[0]
        return deprocess_image(img)    
    
    img_height,img_width=target_size
    #=============================
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
        plt.title('plot_layer_filter(layer_name:%s)'%(layer_name))
        plt.show()
        
#可视化类激活的热力图
def plot_hot_map(model,img_file,layer_name,target_size=(150,150)):
    '''可视化类激活的热力图
    @param model            已经训练好的网络模型
    @param img_file    [str ]测试图像文件
    @param layer_name  [str ]输出卷积层名称，通常为最后一层
    @param target_size [(img_height,img_width)] 图像输出尺寸

    使用范例：
        model=...                                #创建/加载模型/模型编译/模型训练 
        img_file='./test/1.jpg'                  #图像文件
        plot_hot_map(model,layers_name)
    '''
    #图像预处理
    def preprocess_input(x):
        '''
        #圖片預處理使用Keras applications 的 preprocess_input
        #https://medium.com/@sci218mike/%E5%9C%96%E7%89%87%E9%A0%90%E8%99%95%E7%90%86%E4%BD%BF%E7%94%A8keras-applications-%E7%9A%84-preprocess-input-6ef0963a483e        
        
        目前為止Keras提供的pre-train model有
        Xception、VGG16、VGG19、ResNet50、InceptionV3、InceptionResNetV2、MobileNet、DenseNet、NASNet、MobileNetV2
        都可以使用preprocess_input
        (按照Keras documentation排序)
                             input_size        Data_format                    mode
        ----------------------------------------------------------------------------                                 
        Xception             224x224           channels_first/channels_last   tf  
        VGG16                224x224           channels_first/channels_last   caffe
        VGG19                224x224           channels_first/channels_last   caffe
        ResNet50             224x224           channels_first/channels_last   caffe
        InceptionV3          299x299           channels_first/channels_last   tf
        InceptionResNetV2    299x299           channels_first/channels_last   tf
        MobileNet            224x224           channels_last                  tf 
        DenseNet             224x224           channels_first/channels_last   torch
        NASNet               331x331/224x224   channels_first/channels_last   tf
        MobileNetV2          224x224           channels_last                  tf
        ----------------------------------------------------------------------------                                 
        
        各種pretrain model比較
        使用 preprocess_input時輸入為皆為RGB values within [0, 255]

        圖片預處理方式有三種caffe、tf、torch:
        caffe : VGG16、VGG19、ResNet50
        tf : Xception、InceptionV3、InceptionResNetV2、MobileNet、NASNet、MobileNetV2
        torch : DenseNet
        mode = caffe 
        (will convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset)
        減去ImageNet平均 BGR [103.939, 116.779, 123.68]
        mode = tf 
        ( will scale pixels between -1 and 1 )
        除以127.5，然後減 1。
        mode = torch 
        ( will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset)
        除以255，減去ImageNet平均[0.485, 0.456, 0.406] ，除以標準差[0.229, 0.224, 0.225]。        
        '''
        return x/127.5 -1

    #预测值解码
    def decode_predictions(preds,class_indices_rev, top=3):
        print(preds)
        print(class_indices_rev)
        y_index=np.argmax(preds[:top],axis=-1)
        print(y_index)
        y=[class_indices_rev[n] for n in y_index[:]]
        return y    

    img_height,img_width=target_size
    #---------------图像预测----------------
    # The local path to our target image
    # `img` is a PIL image of size 224x224
    img = image.load_img(img_file, target_size=(img_height, img_width)) #加载测试图像

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = image.img_to_array(img)  #转为numpy格式

    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0) #增加samples维

    # Finally we preprocess the batch
    # (this does channel-wise color normalization)
    x = preprocess_input(x)  #图像数组预处理

    preds = model.predict(x) #模型预测
    #class_indices_rev={v:k for k,v in test_gen.class_indices.items()}
    #print('Predicted:', decode_predictions(preds,class_indices_rev, top=3))
    print('preds:',preds.shape)
    
    #---------------显示原始图像--------------
    plt.figure()
    plt.imshow(img)
    plt.title('plot_hot_map:source image')
    plt.show()     
    
    #---------------通过预测值找到对应的输出层(Dense层输出置信度)-------------
    pred_index=np.argmax(preds[0])
    # This is the "african elephant" entry in the prediction vector    
    african_elephant_output = model.output[:, pred_index]

    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    #last_conv_layer_name=layers_name[-1]
    last_conv_layer_name=layer_name
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(last_conv_layer.output.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)   
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap) #热力图
    plt.title('plot_hot_map:heatmap')
    plt.show()    
    
    import cv2
    # We use cv2 to load the original image
    img = cv2.imread(img_file)

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
    ##img_hot_path='%s/work/temp/test_hot.jpg'%(os.getenv('HOME'))
    ##cv2.imwrite(img_hot_path, superimposed_img)
    img=superimposed_img
    fmax=np.max(img)
    fmin=np.min(img)
    img=((img-fmin)/(fmax-fmin))*255
    img=img.astype(np.uint8)

    ##img = image.load_img(img_hot_path, target_size=(img_height, img_width))
    plt.figure()
    plt.imshow(img) #热力图与原始图合成
    plt.title('plot_hot_map:src-heatmap')
    plt.show()  


if __name__=='__main__':
    import os
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    import numpy as np
    #加载模型
    model = VGG16(weights='imagenet', include_top=True)
    #打印模型
    model.summary()
    #测试图像
    img_file='%s/work/data/1.jpg'%(os.getenv('HOME'))
    #输出卷积层
    layers_name=['block1_conv1','block1_conv2']

    #绘制FeatureMap
    plot_FeatureMap(model,layers_name,img_file,target_size=(224,224))
    #绘制网络Filter
    plot_layer_filter(model,layers_name,target_size=(224,224))
    #绘制热力图
    plot_hot_map(model,img_file,'block1_conv2',target_size=(224,224))

