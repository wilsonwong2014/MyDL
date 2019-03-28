"""YOLO_v3 Model Defined in Keras."""
'''
keras.layers.Conv2D
    __init__(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    )
'''

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    '''二维卷积层
        strides 卷积步长，由参数 kwargs传入，默认为(1,1)，即不改变输入大小
        padding 卷积模式,'valid':过滤后输出尺寸变小，'same':过滤后输出尺寸不变
    @param *args 可变参数:(num_filter,filter_size)
    '''
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs) #strides默认(1,1)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    '''封装 激活函数，批正则化，卷积层
        strides 由参数 kwargs传入，默认为(1,1)，即不改变输入大小
    '''
    #深入理解Batch Normalization批标准化
    #   https://www.cnblogs.com/guoyaohua/p/8724433.html
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(), 
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    '''残差层封装模块
    @param x 输入层(上一层输出) ,x.shape=>(batch_size,height,width,old_num_filters)
    @param num_filters 过滤器数目
    @param num_blocks  残差子块个数

    @return 模块输出
        输出大小改变,strides=(2,2)
    '''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)                              #左上角补0
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x) #卷积：改变输出大小
    #x.shape=>(batch_size,height/2,width/2,num_filters)
    #残差子模块
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),       #降低1半过滤器
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)       #恢复过滤器数
        x = Add()([x,y])                                             #残差
    #x.shape=>(batch_size,height/2,width/2,num_filters)
    #y.shape=>(batch_size,height/2,width/2,num_filters)
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    '''darknet网络
    @param x 模型输入,x.shape(32,416,416,3)

    @return x
        x.shape=>(32,13,13,1024)
    '''
    #x.shape=>(32,416,416,3)
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)  #strides=(1,1)
    #x.shape=>(32,416,416,32)
    x = resblock_body(x, 64, 1)               #strides=(2,2)
    #x.shape=>(32,208,208,64)
    x = resblock_body(x, 128, 2)              #strides=(2,2)
    #x.shape=>(32,104,104,128)
    x = resblock_body(x, 256, 8)              #strides=(2,2)  =>y3
    #x.shape=>(32,52,52,256)
    x = resblock_body(x, 512, 8)              #strides=(2,2)  =>y2
    #x.shape=>(32,26,26,512)
    x = resblock_body(x, 1024, 4)             #strides=(2,2)  =>y1
    #x.shape=>(32,13,13,1024)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    '''构建最后输出层y
    @param x 模块输入,x.shape=>(batch_size,height,width,old_num_filter)
    @param num_filters 反馈输出过滤器个数
    @param out_filters 输出层过滤器个数

    @return x,y
        x 反馈给下一个y, x.shape=>(batch_size,height,width,num_filters)
        y 当前构造输出层,y.shape=>(batch_size,height,width,out_filters)
    '''
    #strides=(1,1),因此FeatureMap输出大小不变
    #x.shape=>(batch_size,height,width,old_num_filters)
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    #x.shape=>(batch_size,height,width,num_filters)

    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    #y.shape=>(batch_size,height,width,out_filters)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    '''
    @param inputs      模型输入 => Input(shape=(None, None, 3))
    @param num_anchors 每个输出层[y1,y2,y3]的锚点数,=>3
    @param num_classes 检测类别数，=>80

    @return Model(inputs, [y1,y2,y3])
        y1,y2,y3:num_filters相同,FeatureMap缩小倍数分别为(32,16,8)
        make_last_layers不改变FeatureMap大小
    '''

    '''
    print('============yolo_body==========')    
    xx = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            )(x)
    xxx = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    xxxx = Concatenate()([xxx,darknet.layers[152].output])
    print('x.shape:',x.shape)                                                   #(?,?,?,512)
    print('xx.shape:',xx.shape)                                                 #(?,?,?,256)
    print('xxx.shape:',xxx.shape)                                               #(?,?,?,256)    
    print('xxxx.shape:',xxxx.shape)                                             #(?,?,?,768)
    print('darknet.layers[152].output.shape:',darknet.layers[152].output.shape) #(?,?,?,512)   
    print('============yolo_body{end======')    
    '''

    darknet = Model(inputs, darknet_body(inputs))

    #.darknet.output.shape=>(32,13,13,1024)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5)) #y1输出大小比模型输入缩小2^5=32倍
    # x.shape=>(32,13,13,512)
    #y1.shape=>(32,13,13,num_anchors*(num_classes+5)=255)

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x) #上采样对齐layers[152].output
    #x.shape=>(32,26,26,256)
    x = Concatenate()([x,darknet.layers[152].output])              #keras.layers.Concatenate(axis=-1);以最后一维链接张量
    #darknet.layers[152].output.shape=>(32,26,26,512)
    #                         x.shape=>(32,26,26,768)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))  #y2输出大小比模型输入缩小2^4=16倍
    # x.shape=>(32,26,26,256)
    #y2.shape=>(32,26,26,255)

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x) #上采样对齐layers[92].output
    #x.shape=>(32,52,52,128)
    x = Concatenate()([x,darknet.layers[92].output])
    #darknet.layers[92].output.shape=>(32,52,52,256)
    #                        x.shape=>(32,52,52,384)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))              #y3输出大小比模型输入缩小2^3=8倍
    # x.shape=>(32,52,52,128)
    #y3.shape=>(32,52,52,255)
    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    '''解析模型输出层 y1,y2或y3
        公式推导参考损失函数 yolo_loss
    @param feats       模型输出层y1,y2或y3 ,shape=>(batch_size,height,width,25)
                       feats[...,0:2]=>(x,y),预测原始值为grid的偏移量
                       feats[...,2:4]=>(w,h),预测原始值为与anchor大小比值的对数值
                       以上预测值的模式与yolo_loss相关.
    @param anchors     通过掩码提取的锚点序列(有3个) ,shape=>(3,2)
    @param num_classes 检测类别数(20)
    @param input_shape 模型输入大小(416,416)
    @param calc_loss   是否计算loss,训练模式时为True
    
    @return box_xy,box_wh,box_cofidence,box_class_probs
        box_xy          边框中心坐标预测修正值,全图大小相对值，0-1之间  ,shape=>(batch_size,height,width,num_anchors,2 )
        box_wh          边宽高度和高度预测修正值，全图大小比例值，0-1之间,shape=>(batch_size,height,width,num_anchors,2 )
        box_confidence  预测得分数预测修正值     ,shape=>(batch_size,height,width,num_anchors,1 )
        box_class_probs 检测类别预测修正值       ,shape=>(batch_size,height,width,num_anchors,20)
    '''
    num_anchors = len(anchors) #3
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    
    '''
    feats.shape=>(batch_size,height,width,255) ; 3*(80+5)=255
    grid_shape:(height,width)
    grid_y.shape=>(height,width,1,1)
    grid_x.shape=>(height,width,1,1)
    grid.shape  =>(height.width,1,2)
    '''
    # height, width; feats.shape=>(batch_size,height,width,num_filters); grid_shape:(height,width)
    grid_shape = K.shape(feats)[1:3] 
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    #grid的偏移量 => 全图大小的相对偏移量,0-1之间
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    #与anchor大小比值的对数值 => 全图大小比例值,没有做限值(0-1)处理，预测宽高有可能大于原图大小，在后续处理中做边界处理
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))  
    box_confidence = K.sigmoid(feats[..., 4:5]) #限值0-1之间
    box_class_probs = K.sigmoid(feats[..., 5:]) #限值0-1之间

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    '''还原原始图像真实物理大小
    @param box_xy      预测值,相对全图大小偏移量,0-1之间,shape=>(batch_size,height,width,num_anchors,2)
    @param box_wh      预测值,相对全图大小比例值，       shape=>(batch_size,height,width,num_anchors,2)
    @param input_shape 模型输入尺寸,val=>(416,416)
    @param image_shape 图像真实尺寸,shape=>(?,?)
    @return boxes
    '''
    #归一化数据状态
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))                #模型输入大小
    image_shape = K.cast(image_shape, K.dtype(box_yx))                #图像真实大小
    new_shape = K.round(image_shape * K.min(input_shape/image_shape)) #原始图像等比缩放后的shape
    offset = (input_shape-new_shape)/2./input_shape                   #原始图像等比缩放后与模型输入shape的边偏移量(归一化)
    scale = input_shape/new_shape                                     #图像缩放倍数
    #真实数据复原,box_yx-offset为get_random_data的box[:, [0,2]] = box[:, [0,2]]*scale + dx的逆向操作
    box_yx = (box_yx - offset) * scale 
    box_hw *= scale                                                   #真实数据复原

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1], # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    #还原真实物理尺寸
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    '''处理卷积层的输出[y1,y2或y3]
    @param feats       y1,y2或y3,[(13,13,3,85),(26,26,3,85),(52,52,3,85)]
    @param anchors     通过掩码提取的锚点序列(3个)
    @param num_classes 检测类别数(80)
    @param input_shape 模型输入大小(416,416)
    @param image_shape 图像原始尺寸

    @return boxes,box_scores
        box_xy:(n,2)
        box_wh:(n,2)
        box_confidence:(n,1)
        box_class_probs:(n,80)
    '''
    #提取预测边框
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    #修正预测边框为真实值
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape) #=> (ymin,xmin,ymax,xmax)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,       #由yolo_body定义输出[y1,y2,y3]
              anchors,            #shape:(9,2)
              num_classes,        #80
              image_shape,        #Tensor:(2,),图像真实大小[h x w]
              max_boxes=40,       #最大预测外框数
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    '''YOLO模型评估，给予输入，返回预测结果
        anchors分配策略：一共有9个anchor,三个模型输出[y1,y2,y3]
        anchor从小到大排序为0,1,2,3,4,5,6,7,8
        y1的分辨率最小(13x13),其次为y2的分别率为(26x26),y3的分辨率最高(52x52)
        高分辨率更容易检测小物体，因此，小的anchor分配给高分辨率的输出，大的anchor分配给小分辨率的输出
        anchors(0,1,2)=>y3
        anchors(3,4,5)=>y2
        anchors(6,7,8)=>y1
    @param yolo_outputs 模型输出[y1,y2,y3]，在yolo_body函数定义.
    @param anchors 锚点数组(9 x 2)
    @param num_classes 检测类别数(80)
    @param image_shape 图像真实大小张量(2,1)
    @param max_boxes 最大预测边框数(20)
    @param score_threshold 返回预测结果的得分阈值(0.6)
    @param iou_threshold 非极值抑制参数(0.5)
    
    @return boxes_,scores_,classes_
        boxes_  :边框预测值  ,shape=>(?,4)=>[top, left, bottom, right]
        scores_ :得分(置信度),shape=>(?,1)
        classes_:类别ID,      shape=>(?,1)
    '''
    print('===============yolo_eval==============')
    num_layers = len(yolo_outputs) #3:y1,y2,y3
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32  #三个输出层的缩放比例分别为32,16,8；因此仅需其中一个层就可以获取模型输入大小
    boxes = []
    box_scores = []

    #==============提取预测结果：bbox,score===============
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)           #第一维拼接
    box_scores = K.concatenate(box_scores, axis=0) #第一维拼接

    #========条件条件过滤：得分 && 非极大值抑制============
    mask = box_scores >= score_threshold                    #得分条件过滤
    max_boxes_tensor = K.constant(max_boxes, dtype='int32') #最大预测数 40
    print('boxes.shape:',boxes.shape)                       #boxes.shape: (?, 4)
    print('boxe_score.shape:',box_scores.shape)             #boxe_score.shape: (?, 80)
    print('mask.shape:',mask.shape)                         #mask.shape: (?, 80)
    boxes_   = []
    scores_  = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])                 #阈值条件过滤
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c]) #阈值条件过滤
        '''非极大值抑制
        https://www.cnblogs.com/makefile/p/nms.html
        原理：对于Bounding Box的列表B及其对应的置信度S,采用下面的计算方式.选择具有最大score的检测框M,
            将其从B集合中移除并加入到最终的检测结果D中.通常将B中剩余检测框中与M的IoU大于阈值Nt的框从B中移除.重复这个过程,直到B为空.
        '''
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_   = K.concatenate(boxes_, axis=0)
    scores_  = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    '''处理true_boxs为训练提供数据输入
        模型输入数据形状:[image_data:(bm,h,w,3),y1:(bm,13,13,3,25),y2:(bm,26,26,3,25),y3:(bm,52,52,3,25)]
        iamge_data:已做数据增强处理的归一化图像数据
        y1,y2,y3:标注数据，已作归一化处理,每个grid蕴含3个预测框、1个obj预测,20个分类预测
        要点说明：
          1.通过kmean从数据集选取9个anchors.
          2.9个anchors按预设规则分配给[y1,y2,y3]中每块的3个预测框
          3.一个样本有N个有效标注(前面N个).
          4.每个标注bbox与9个anchors做IoU计算，并选取最大值，在对应的分配预测框设置标注数据(bbox+clssid)
          5.归纳：为9个anchors分配合适的预测框索引[y1,y2,y3][0,1,1]，
              计算标注bbox与9个anchors的IoU并选最大值对应的预测框索引设值(bbox,classid)
    @param true_boxes  shape=(m,T,5),经过缩放、偏移、未归一化处理,前面N个box有效，之后的box无效
    @param input_shape 模型输入大小(416,416)
    @param anchors     锚点数组,shape=>(9,2),wh
    @param num_classes 检测类别数,val=>20
    
    @return [y1,y2,y3]
        y1:shape=>(32,13,13,3,25)
        y2:shape=>(32,26,26,3,25)
        y3:shape=>(32,52,52,3,25)
        x,y,w,h是归一化值，相对值0-1之间
    '''
    #class id 最大为(num_classes-1)
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    #================锚点分组=================
    #锚点数组分为三个子集，分别分配给y1,y2,y3
    num_layers = len(anchors)//3 # default setting,y1,y2,y3
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    #========box范式转换并归一化处理===========
    #[x_min,y_min,x_max,y_max]=>[x,y,w,h]
    true_boxes = np.array(true_boxes, dtype='float32')            #shape=>(32,20,5)
    input_shape = np.array(input_shape, dtype='int32')            #shape=>(2,),val=>[416,416]
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2 #shape=>(32,20,2),((x_min+x_max)/2,(y_min+y_max)/2)
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]        #shape=>(32,20,2),((x_max-x_min)  ,(y_max-y_min)  )
    #归一化处理，(x_min,y_min,x_max,y_max)=>(x,y,w,h)
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]           #归一化处理,input_shape是tuple类型,input_shape[::-1]翻转排序
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]           #归一化处理
    
    #=======构造标注数组 y_true =============
    m = true_boxes.shape[0] #=>batch_size:32
    #grid_shapes=>[(13,13),(26,26),(52,52)]  #y1,y2,y3  #hw
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)] 
    #构造y_true形状[(32,13,13,3,25),(32,26,26,3,25),(32,52,52,3,25)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    #print('len(y_true):',len(y_true))           #=>len(y_true): 3
    #print('y_true[0].shape:',y_true[0].shape)   #=>y_true[0].shape: (32, 13, 13, 3, 25)
    #print('y_true[1].shape:',y_true[1].shape)   #=>y_true[1].shape: (32, 26, 26, 3, 25)
    #print('y_true[2].shape:',y_true[2].shape)   #=>y_true[2].shape: (32, 52, 52, 3, 25)

    #======过滤无效外框=====================
    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0) #在前面扩展一维=>anchors.shape=>(1,9,2)
    anchor_maxes = anchors / 2.          #shape=>(1,9,2)，相对中点右下角偏移
    anchor_mins = -anchor_maxes          #shape=>(1,9,2)，相对中点左上角偏移
    valid_mask = boxes_wh[..., 0]>0      #筛选有效数据,valid_mask.shape=>(32,20),前面N个box有效，之后的box无效

    #===提取图像有效框，匹配最佳anchor，在对应y_true位置填充box,obj,cls=====
    # 1.遍历批数据的每个图像=>m
    # 2.提取图像有效框=>wh
    # 3.计算每个框匹配的最佳anchor(IoU计算)=>n=>k
    #    根据n找到所属层(y1,y2 or y3)和filter_index
    # 4.查找最佳anchor属于那个层(anchor_mask)=>l
    # 5.换算外框中心所属grid cell=>(j,i)
    # 6.提取类别ID=>c
    # 7.填充 y_true
    #       y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]  #x,y,w,h
    #       y_true[l][b, j, i, k, 4] = 1                       #所有检测类别统一视为Object
    #       y_true[l][b, j, i, k, 5+c] = 1                     #对应类别序号上设置标记
    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  #shape=>(?,2); ?-当前样本的有效box个数,相当于提取前面的N个有效box
        if len(wh)==0: continue          #无效box，直接跳过
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)      #shape=>(?,1,2),扩展一维
        box_maxes = wh / 2.              #shape=>(?,1,2)，相对中点右下角偏移
        box_mins = -box_maxes            #shape=>(?,1,2)，相对中点左上角偏移
        
        #IOU计算:一个图像有?个box，每个box与9个anchors做IOU计算，并取最大值作为输出
        #        ?个box与9个anchors交叉做IOU计算
        #box_mins.shape=>(?,1,2),anchor_mins.shape=>(1,9,2)
        intersect_mins = np.maximum(box_mins, anchor_mins)               #shape=>(?,9,2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)            #shape=>(?,9,2)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)  #shape=>(?,9,2)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]     #shape=>(?,9)
        box_area = wh[..., 0] * wh[..., 1]                               #shape=>(?,1)
        anchor_area = anchors[..., 0] * anchors[..., 1]                  #shape=>(1,9)
        iou = intersect_area / (box_area + anchor_area - intersect_area) #shape=>(?,9)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)                            #shape=>(?,),每个box只选取一个最大anchor输出

        #通过遍历每个box的best_anchor在对应的维度设置y_true
        for t, n in enumerate(best_anchor): #t:box index,n:anchor index
            for l in range(num_layers):     #y1,y2,y3
                #查找best anchor所在anchor mask,在[y1,y2,y3]中总会找到对应的ancho index
                if n in anchor_mask[l]:     #anchor分配策略：[[6,7,8], [3,4,5], [0,1,2]]
                    #true_boxes.shape=>(batch_size,max_boxs,5):[x,y,w,h,cls_id]
                    #x对应的单元格(如,13 x 13)的col_index
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')  
                    #y对应的单元格(如,13 x 13)的row_index
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')  
                    k = anchor_mask[l].index(n)                                        #pred box index:0,1,2
                    c = true_boxes[b,t, 4].astype('int32')                             #class_id
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]                  #x,y,w,h
                    y_true[l][b, j, i, k, 4] = 1                                       #所有检测类别统一视为Object
                    y_true[l][b, j, i, k, 5+c] = 1                                     #对应类别序号上设置标记

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    '''损失函数
        https://blog.csdn.net/jesse_mx/article/details/53925356
        标注数据y_true的x,y,w,h说明：
        1. x,y数据说明：
          1.1. 原始数据x,y为全图的相对偏移量，值范围在0-1之间
          1.2. x,y在yolo_loss修正为grid的偏移量，值范围在0-1之间
          1.3. 因此网络输出的预测值y_pred为grid的偏移量，值范围在0-1之间
          1.4. 修正/反修正过程
               修  正：全图相对偏移量=>grid偏移量
                  raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
               反修正:grid偏移量=>全图相对偏移量
                  feats[...,:2] --- 预测值 x,y；grid偏移量
                  grid ------------ 单元格坐标值:整数
                  K.sigmoid(feats[...,:2]) --- 把预测值限定在0-1之间
                  K.sigmoid(feats[...,:2])+grid --- 把预测值修正为绝对偏移量
                  box_xy=(K.sigmoid(feats[...,:2])+grid)/K.cast(grid_shape[::-1],K.dtype(feats)) 
                        ---把预测值修正为grid_shape的相对偏移
        2. w,h数据说明：
          2.1. 原始数据w,h为全图的相对偏移量，值范围在0-1之间
          2.2. w,h在yolo_loss修正为与anchor大小比例的对数值
          2.3. 因此网络输出的预测值y_pred为anchor大小比例的对数值
          2.4. 修正/反修正过程
               修  正:全图相对偏移量=>与anchor大小比例的对数值
                    raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])                
               反修正:与anchor大小比例的对数值=>全图相对偏移量
                    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))  
        
    @param args    [list ]Lambda层输入, [*model_body.output, *y_true]=>
                                        [y1_pred, y2_pred, y3_pred,y1_true, y2_true, y3_true]
                                        [(?,?,75),(?,?,75),(?,?,75),(?,13,13,3,25),(?,26,26,3,25),(?,52,52,3,25)]
                        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
                            [*model_body.output, *y_true])
    @param anchors        [array]锚点数组,shape=>(9,2)
    @param num_classes    [int  ]检测类别数,val=20
    @param ignore_thresh  [float]
    @param print_loss     [bool ]

    @return loss
    '''
    print('==========yolo_loss===========')
    #print([x.shape for x in args])
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers] #预测值, y_pred
    y_true = args[num_layers:]       #真实值, y_true
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    #模型输入大小,val=>(416,416)
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))                         
    #模型输出FeatureMap大小，val=>[(13,13),(26,26),(52,52)]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)] 
    loss = 0 #损失值
    #batch_size
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers): #y1,y2,y3
        object_mask = y_true[l][..., 4:5]     #是否有目标标记，0-没有目标，1-有目标,shape=>(batch_size,height,width,3,1)
        true_class_probs = y_true[l][..., 5:] #分类检测值,shape=>(batch_size,height,width,3,20)

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        '''
        grid.shape  =>(height,width,1,2)
        raw_pred    =>yolo_outputs[l]
        '''
        pred_box = K.concatenate([pred_xy, pred_wh])  #预测边框

        # Darknet raw box to calculate loss.
        #标注数据修正为对grid的偏移
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid                           
        #标注数据修正为对anchor比例的对数值
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1]) 
        #0除处理
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))   # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        #raw_pred   :模型输出y1,y2,y3，未经特别处理
        #raw_true_xy:相对于grid cell的偏移量归一化值
        #raw_true_wh:边框尺寸对anchor的缩放比例
        #object_mask:标记每个grid cell是否有object,0-没有，1-有
        
        #xy的loss
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True) 
        #wh的loss
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])                           
        #IoU的loss
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask      
        #分类loss
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)           

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], 
                            message='loss: ')
    return loss
