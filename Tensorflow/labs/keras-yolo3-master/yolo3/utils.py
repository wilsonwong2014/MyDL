"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    #函数堆叠
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    #不改变图像长宽比例，取新图像的长与旧图像的长，新图像的宽与旧图像的宽比之最小值为比例重设图像大小，居中，空白处以(128,128,128)填充
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    '''读取一个样本数据：
        0.box的x_min,y_min,x_max,y_max是相对原始图像左上角的像素坐标
        1.处理原始图像数据
          1.1.读取原始图像数据
          1.2.对原始图像做缩放、偏移、归一化处理
          1.3.图像变形处理超出部分被裁剪，空缺部分以(128,128,128)填充
        2.同步处理boxes数据
          2.1.同步对boxes做缩放、偏移处理、未归一化
          2.2.边界越界处理:x_min>=0,y_min>=0,x_max<=w,y_max<=h
          2.3.box过滤处理:过滤扭曲变形后在模型输入框之外的box

    @param annotation_line 一个样本数据,形如: img_file BOX1 BOX2 ... BOXN
                            BOX格式: x_min,y_min,x_max,y_max,class_id
                            范例：/data/1.jpg 6,1,314,262,19 40,97,121,411,4 137,36,169,109,14 180,36,216,104,14
    @param input_shape 模型输入尺寸(416,416)
    @param random      是否提供随机化增强数据处理
    @param max_boxes   最大预测外框数
    @param jitter      数据增强，长宽抖动系数,可改变图像的尺寸大小
    @param hue         数据增强，色调
    @param sat         数据增强，饱和度
    @param val         数据增强，明亮度
    @param proc_img    重设图像大小，random=False时有效，内定为True（如果为False,image_data返回None，无意义！）

    @return image_data,box_data
        image_data shape:(416,416,3),原始图像数据增强(缩放、偏移、HSV变换)，归一化处理
        box_data   shape:(20,5)     ,处理后的box数据(缩放、偏移、过滤无效框,未归一化),前面N个box有效，之后的box无效
    '''
    #以空格分离各个子项
    line = annotation_line.split() #=>['img_file', '6,1,314,262,19', '40,97,121,411,4', '137,36,169,109,14', '180,36,216,104,14', '96,39,123,103,14']
    #读取图像数据
    image = Image.open(line[0])    #=>img_file
    #图像尺寸(不固定)
    iw, ih = image.size
    #模型输入尺寸
    h, w = input_shape             #=>(416,416)
    #边框标签值，原始数据,shape:(?,5),[x_min,y_min,x_max,y_max,class_id]
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]) #=>分离bounding box =>[[6,1,314,262,19],[40,97,121,411,4],[137,36,169,109,14],[180,36,216,104,14],[96,39,123,103,14]]

    #===============不对原始图像做增强处理=================
    #图像数据与box数据同步处理:缩放，偏移;预测时需要逆向操作
    if not random:
        #--------------模型输入图像数据处理----------------
        # resize image
        scale = min(w/iw, h/ih) #缩放比例
        nw = int(iw*scale)      #重设图像宽度
        nh = int(ih*scale)      #重设图像高度
        dx = (w-nw)//2          #左右边隙:图像区边界与模型输入框边界距离
        dy = (h-nh)//2          #上下边隙:图像区边界与模型输入框边界距离
        image_data=0
        if proc_img:
            #不改变长宽比例重设图像大小
            image = image.resize((nw,nh), Image.BICUBIC)       #重设图像大小(不改变对比度)
            new_image = Image.new('RGB', (w,h), (128,128,128)) #模型输入图像格式，设置背景颜色为(128,128,128)
            new_image.paste(image, (dx, dy))                   #把重设大小后的原始图像粘帖到居中
            image_data = np.array(new_image)/255.              #模型输入数据：归一化处理

        #--------------边框标签值数据处理------------------
        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes] #最多处理20个box,意味着一张图像最多能预测20个目标，此参数可以修改，但必须在相关代码同步
            box[:, [0,2]] = box[:, [0,2]]*scale + dx     #x坐标处理[缩放，偏移],[x_min,x_max]
            box[:, [1,3]] = box[:, [1,3]]*scale + dy     #y坐标处理[缩放，偏移],[y_min,y_max]
            box_data[:len(box)] = box

        return image_data, box_data

    #=================对原始图像做增强处理==================
    #-------------重设图像大小-------------
    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter) #宽高对比度随机化
    scale = rand(.25, 2)                                           #缩放比例随机化
    if new_ar < 1:                                                 #w < h
        nh = int(scale*h)                                          #新的图像高度
        nw = int(nh*new_ar)                                        #新的图像宽度
    else:                                                          #w > h
        nw = int(scale*w)                                          #新的图像宽度
        nh = int(nw/new_ar)                                        #新的图像高度
    image = image.resize((nw,nh), Image.BICUBIC)                   #重设图像大小,会变形

    #------------图像粘帖----------------
    # place image
    dx = int(rand(0, w-nw))                                        #左边隙:图像区边界与模型输入框边界距离
    dy = int(rand(0, h-nh))                                        #顶边隙:图像区边界与模型输入框边界距离
    new_image = Image.new('RGB', (w,h), (128,128,128))             #模型输入图像格式，设置背景颜色为(128,128,128)
    new_image.paste(image, (dx, dy))                               #把重设大小后的原始图像粘帖
    image = new_image

    #------------图像翻转处理------------
    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    #------------HSV颜色模型随机化-------
    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    #---根据缩放，偏移修正边宽参数值------
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx                 #x坐标处理[缩放，偏移],[x_min,x_max]
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy                 #y坐标处理[缩放，偏移],[y_min,y_max]
        if flip: box[:, [0,2]] = w - box[:, [2,0]]               #图像翻转
        box[:, 0:2][box[:, 0:2]<0] = 0                           #边界处理
        box[:, 2][box[:, 2]>w] = w                               #边界处理  
        box[:, 3][box[:, 3]>h] = h                               #边界处理
        box_w = box[:, 2] - box[:, 0]                            #宽度
        box_h = box[:, 3] - box[:, 1]                            #高度
        box = box[np.logical_and(box_w>1, box_h>1)]              #过滤非法边框 discard invalid box，扭曲变形把box挤出模型输入区域(416x416)
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
