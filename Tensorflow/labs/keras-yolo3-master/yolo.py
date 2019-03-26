# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import tensorflow as tf
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    #检测参数必须要与训练模型相匹配
    _defaults = {
        "model_path": 'model_data/yolo.h5',            
        "anchors_path": 'model_data/yolo_anchors.txt',  #9个锚点
        "classes_path": 'model_data/coco_classes.txt',  #80个分类
        "log_path":'model_data/log/detect',             #日志目录
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        '''构造函数
        @param **kwargs 字典类型参数，支持字段由_defaults={...}定义.
        '''
        print('==============__init__=>kwargs:===========\n')
        for k,v in kwargs.items():
            print('%s:%s'%(k,v))
        print('==============__init__=>kwargs{end}=======\n')

        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate() #构造测试模型=>返回预测边框，得分，检测类型

    def _get_class(self):
        '''获取检测分类列表
        注意：测试与训练配置必须保持一致!
        分类列表存放于文件 self.classes_path,每行一个分类名称，如：
            dog
            cat
            ...
        返回值：分类名称list,如：['dog','cat',...]
        '''
        #展开用户路径，如：os.path.expanduser('~/data')=>/home/hjw/data
        classes_path = os.path.expanduser(self.classes_path)    
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        '''获取锚点数组
        注意：测试与训练必须保持一致！
        锚点数组数值存放于文件 self.anchors_path，锚框长宽由两个数值描述，多个锚框连续存放在一行，逗号隔开，如：第一值为高度，第二值为宽度
            13，13，45，60，80,200,......
        返回值：[9 x 2]数组，每行表示一个锚框长宽，第一列为宽度，第二列为高度
        '''
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)  #返回 n x 2,每行是一个锚点

    def generate(self):
        '''构造测试模型
        内置输入参数
            model.input: image_data,修正尺寸后的图像数据
            input_image_shape: [image.size[1],image.size[0]],
                    原始图像尺寸;由yolo_val输入张量(2,)，经session.run(feed_dict={self.input_image_shape:...})填充输入值 
            K.learning_phase():0,测试非训练标记
            anchors:锚点数组,[9 x 2],第一列为宽度，第二列为高度
            class_names:检测类别list[80 x 1]
            score_threshold:Bounding Box预测得分阈值
            iou_threshold:IOU阈值
        
        @return boxes, scores, classes
            boxes.shape  =>(?,4)  ,修正后的真实原始图像边框值,(ymin,xmin,ymax,xmax)
            scores.shape =>(?, )
            classes.shape=>(?, )
        '''
        print('============generate=============')
        model_path = os.path.expanduser(self.model_path)  #模型参数文件
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        #======================加载模型===================
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False) #加载模型
            #self.yolo_model.summary(line_length=250)               #打印模型参数
            self.yolo_model.summary()                               #打印模型参数
            #输出日志
            summary_writer = tf.summary.FileWriter(self.log_path, tf.get_default_graph())
            summary_writer.close()
        except:
            #创建模型
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            #加载参数
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            #模型参数与anchor,class不匹配
            print('self.yolo_model.layers[-1].output_shape[-1]:',self.yolo_model.layers[-1].output_shape[-1])
            print('num_anchors/len(self.yolo_model.output) * (num_classes + 5):',num_anchors/len(self.yolo_model.output) * (num_classes + 5))
            print('num_anchors:',num_anchors)
            print('len(self.yolo_model.output):',len(self.yolo_model.output))
            print('num_classes:',num_classes)
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, )) #检测图像真实尺寸张量
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        print('self.yolo.model.output.shape:')
        #此范例：anchors=9,num_classes=80,yolo_model.output为[y1,y2,y3],平均每个分配3个anchor => 9/3*(80+5)=255)
        print([x.shape for x in self.yolo_model.output])       #[(none,none,none,255)
                                                               #,(none,none,none,255)
                                                               #,(none,none,none,255)]
        print('self.anchors:',self.anchors)                    #(9,2)
        print('self.class_names:',self.class_names)            #(80,1)
        print('self.input_image_shape:',self.input_image_shape)#(2,1)
        print('self.score:',self.score)
        print('self.iou:',self.iou)

        #==================yolo评估====================
        boxes, scores, classes = yolo_eval(
                    self.yolo_model.output       #模型输出:[y1,y2,y3]
                    , self.anchors               #锚点数组:[9 x 2]
                    , len(self.class_names)      #分类数目：80
                    , self.input_image_shape     #原始图像大小:张量,placeholder,模型输入
                    , score_threshold=self.score #得分阈值
                    , iou_threshold=self.iou     #交并比阈值
                    )
        print('boxes.shape:',boxes.shape)                      #=>boxes.shape: (?, 4),回归框:[x1,y1,x2,y2]
        print('scores.shape:',scores.shape)                    #=>scores.shape: (?,) ,得分:[0-1]
        print('classes.shape:',classes.shape)                  #=>classes.shape: (?,),预测类别
        return boxes, scores, classes

    def detect_image(self, image,draw=True):
        '''图像目标检测
            在图像上检测目标，绘制检测结果,返回检测结果(外框、类别名、置信度绘制在原始图像上).
        @param image 图像数组(image = Image.open(img_file))，原始数据，没有经过任何预处理
        @param draw  是否绘制检测结果
        @return 
            if draw=True
              return 处理后的image(外框、类别名、置信度绘制在原始图像上)
            else
              return (out_boxes, out_scores, out_classes)
        '''
        start = timer()

        #======图像数据预处理：修正为标准尺寸(self.model_image_size)，归一化处理，扩展维度=========
        #letterbox_image:不改变比例重置图像大小
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32') #修正尺寸后的图像数据，修正尺寸为了适应模型

        #print(image_data.shape)
        image_data /= 255.                          #归一化处理
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        time1=timer()
        #=============执行评估模型=>yolo_eval(...)=================
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,                     #模型标准化输入数据:(1,416,416,3)
                self.input_image_shape: [image.size[1], image.size[0]],#原始图像大小:(height,width)
                K.learning_phase(): 0                                  #测试模式
            })
        time2=timer()
        #=========================结果返回========================
        if not draw:
            #返回检测结果数据(非绘制)
            out_classesname = [self.class_names[c] for c in out_classes]
            return out_boxes,out_scores,out_classes,out_classesname
        else:
            #返回检测结果图(绘制)
            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            #提取结果并绘制外框
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]              #预测类别
                box = out_boxes[i]                                 #预测外框
                score = out_scores[i]                              #得分值  

                label = '{} {:.2f}'.format(predicted_class, score) #外框标题
                draw = ImageDraw.Draw(image)                       #加载原始图像到画布
                label_size = draw.textsize(label, font)            #分类框标题尺寸大小

                top, left, bottom, right = box                     #外框坐标:(y1,x1,y2,x2)
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                #标题绘制坐标始点
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                #绘制检测目标外框
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                #绘制标题外框
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                #绘制标题
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            end = timer()
            print('end - start:',end-start)
            print('time2 - start:',time2-start)
            print('time2 - time1:',time2-time1)
            return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

