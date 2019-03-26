"""
Retrain the YOLO model for your own dataset.
"""
import pdb
pdb.set_trace()
import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

path='%s/data/VOCdevkit/VOC2012/model_data'%(os.getenv('HOME')) #数据存放根目录

def _main():
    #===============训练参数配置==================
    annotation_path = '%s/train.txt'%(path)      #训练样本，注释文件,格式：/img_file 6,1,314,262,19 40,97,121,411,4 137,36,169,109,14 180,36,216,104,14 96,39,123,103,14
                                                 #由voc_annotation.py生成
    log_dir = '%s/logs/000/'%(path)              #日志目录
    classes_path = '%s/voc_classes.txt'%(path)   #VOC数据集标签类别(20类)
    anchors_path = '%s/yolo_anchors.txt'%(path)  #yolo anchor配置文件[10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    class_names = get_classes(classes_path)      #读取VOC数据集标签类别
        #class_names=>[aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor]
    num_classes = len(class_names)               #标签类别数(20)
    anchors = get_anchors(anchors_path)          #锚点wh,anchors [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
    input_shape = (416,416)                      # multiple of 32, hw #模型输入尺寸

    #===============创建训练模型=================
    #freeze_body=2 => 冻结除y1,y2,y3的所有层
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='%s/tiny_yolo_weights.h5'%(path))
    else:
        #freeze_body:1-解冻所有层，2-冻结darknet
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='%s/yolo_weights.h5'%(path)) # make sure you know what you freeze
        print('mdoel.input_shape:',model.input_shape)   #mdoel.input_shape: [(None, None, None, 3), (None, 13, 13, 3, 25), (None, 26, 26, 3, 25), (None, 52, 52, 3, 25)]
        print('mdoel.output_shape:',model.output_shape) #mdoel.output_shape: (None, 1) --- loss
    
    #================训练回调函数设置=================
    #TensorBoard可视化日志
    logging = TensorBoard(log_dir=log_dir)
    #断点保存
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    #学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    #退出条件
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #=================加载训练样本文件================
    val_split = 0.1                     #留出百分之10的数据用于校验
    with open(annotation_path) as f:
        lines = f.readlines()           #每行形如以下格式：img_file 6,1,314,262,19 40,97,121,411,4 137,36,169,109,14 180,36,216,104,14 96,39,123,103,14
    np.random.seed(10101)               #设置随机种子，固化每次的随机序列(伪随机)
    np.random.shuffle(lines)            #打乱顺序
    np.random.seed(None)                #恢复随机状态
    num_val = int(len(lines)*val_split) #校验样本数
    num_train = len(lines) - num_val    #训练样本数

    #=====================冻结darknet调参=======================
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        #=================模型编译===================
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        #=================构造训练数据生成器==========
        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        #anchors=>[[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
        data_gen=data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
        print(type(data_gen))
        #print('data_gen.shape:',data_gen.shape) #
        #data_gen.shape=>([(32, 416, 416, 3), (32, 13, 13, 3, 25), (32, 26, 26, 3, 25), (32, 52, 52, 3, 25)],(32,))

        #=================模型训练====================
        model.fit_generator(data_gen,
                steps_per_epoch=1,#steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=1, #validation_steps=max(1, num_val//batch_size),
                epochs=1,#epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])

        #=================保存训练参数================
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')


    #=====================解冻所有层调参=========================
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        #==============模型层解冻==============
        start_layer=45  #从那个层开始解冻，原配置为0,但出现内存溢出错误！
        for i in range(start_layer,len(model.layers)):
            model.layers[i].trainable = True

        #==============模型编译================
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze [%d:%d] of the layers.'%(45,len(model.layers)))

        #==============模型训练================
        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=1,#steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=1,#validation_steps=max(1, num_val//batch_size),
            epochs=2,#epochs=100,
            initial_epoch=1,#initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping]
            )

        #=============保存模型参数============
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    '''加载检测类别名称
    @param classes_path 检测类别文件,文件内容每行表示一个类别名称，如：
        dog
        cat
    @return class_names [list]检测类别名称
    '''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    '''加载锚点数组
    @param anchors_path 锚点文件路径,文件内容如下所示：
        10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
        每两个组成一个锚点，标识边框的宽度与高度wh
    @return np.array(anchors).reshape(-1, 2) 
    '''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    '''构建模型
    @param input_shape     [tuple]模型输入尺寸,val=>(416,416)
    @param anchors         [array]锚点数组,shape=>(9,2)
    @param num_classes     [int  ]检测类别数,val=>20
    @param load_pretrained [bool ]是否预加载参数
    @param freeze_body     [int  ]模型层的冻结方式，1-冻结darknet53,2-除y1y2y3的所有层，其他-不做冻结设置
    @param weights_path    [str  ]预训练模型参数路径

    @return model
        模型输入:image_data,y1,y2,y3
        模型输出:loss
    '''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    #=============定义输出标签值:y_true=[y1,y2,y3]=============
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
    '''
    h=>416
    w=>416
    num_anchors=>9
    num_classes=>20
    y_true=>[<tf.Tensor 'input_1:0' shape=(?, 13, 13, 3, 25) dtype=float32>,
             <tf.Tensor 'input_2:0' shape=(?, 26, 26, 3, 25) dtype=float32>,
             <tf.Tensor 'input_3:0' shape=(?, 52, 52, 3, 25) dtype=float32>]
    '''
    
    #=============yolo主体模型================================
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    #=============加载预训练模型参数==========================
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    #=============定义网络loss===============================
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])

    #=============重构模型===================================
    model = Model([model_body.input, *y_true], model_loss)

    #输出最终训练模型
    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    '''数据生成器
    @param annotations_lines 训练样本，每行形如以下格式：img_file 6,1,314,262,19 40,97,121,411,4 137,36,169,109,14 180,36,216,104,14 96,39,123,103,14 
    @param batch_size 批大小,32
    @param input_shape 模型输入尺寸(416,416)
    @param anchors bounding box 锚点wh:[[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
    @param num_classes 标签类别数,20
    @return yield [image_data, *y_true], np.zeros(batch_size) => ([image_data,y1,y2,y3],loss)
                                                              => ([(32,416,416,3),(32,13,13,3,25),(32,26,26,3,25),(32,52,52,3,25),(32,)])
    '''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0: #一轮
                np.random.shuffle(annotation_lines) #打乱排序
            #生成一个样本数据,image.shape=>(416,416,3),box.shape=>(20,5)
            #image做数据增强、归一化处理；box与image同步做变形、偏移处理，未归一化
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)  
            image_data.append(image) #图像序列
            box_data.append(box)     #bounding box序列
            i = (i+1) % n
        image_data = np.array(image_data)                                           #=>shape:(32,416,416,3)
        box_data = np.array(box_data)                                               #=>shape:(32,20,5)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes) #=>把样本原始数据转换为训练需要的数据格式
        '''
        image_data:shape=>(32,416,416,3),模型输入图像数据
        y_true:[y1,y2,y3],               模型输入Box标签数据
            y1:shape=>(32,13,13,3,25)
            y2:shape=>(32,26,26,3,25)
            y3:shape=>(32,52,52,3,25)
        np.zeros(batch_size):            模型输出loss

        yield [image_data, *y_true], np.zeros(batch_size) => ([image_data,y1,y2,y3],loss)
                                                          => ([(32,416,416,3),(32,13,13,3,25),(32,26,26,3,25),(32,52,52,3,25),(32,)])
        '''        
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main() #训练函数入口
