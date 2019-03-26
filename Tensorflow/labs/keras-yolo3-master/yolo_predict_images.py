import pdb
pdb.set_trace()
import sys
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import pandas as pd
from mylibs.ProcessBar import ShowProcess

def detect_img(arg,yolo):
    '''批量图像目标检测，把结果保存在数据文件
    @param arg  参数
        arg.JPEGImages ----图片目录
        arg.imagesetfile --测试图片集,文本文件，每行标识一个图片名称(不含扩展名) ,如：
                2008_00001
                2008_00002
        arg.resultfile ----保存结果文件,CSV格式，第一行为字段名，其余每行为一条记录，如：
            ,filename,score,x,y,w,h,classid,classname
           0,2008_000001,0.987,10,20,30,50,0,dog
           1,2008_000001,0.78 ,30,50,200,100,1,cat
    @param yolo YOLO对象,对象构造，如：
        yolo=YOLO({'score'：0.3,'iou':0.5}) #详细参数支持见YOLO实现代码 _default={...}
    '''
    images_path=arg.JPEGImages       #图片目录
    images_set_file=arg.imagesetfile #测试图片集
    results_file=arg.resultfile      #结果输出文件
    print('images_path:',images_path)
    print('images_set_file:',images_set_file)
    print('results_file:',results_file)

    #进度条
    max_steps = 100
    process_bar = ShowProcess(max_steps,'','', 'OK') 

    #读取测试图片集
    with open(images_set_file,'r') as f:
        lines=f.readlines()
    imagenames = [x.strip() for x in lines]
    num_images=len(imagenames)  #测试图片集数量

    #用pd.DataFrame保存结果
    df=pd.DataFrame(columns=['filename','score','top','left','bottom','right','classid','classname'],dtype=np.float32)
    key_index=0 #关键字值(整型递增)

    #遍历检测图片
    for i,sname in enumerate(imagenames):
        sfile=os.path.join(images_path,'%s.jpg'%(sname))
        image=Image.open(sfile)
        #图片检测
        out_boxes, out_scores, out_classes,out_classesname=yolo.detect_image(image,draw=False)
        #检测结果解析，并向DataFrame添加一行
        for n in range(len(out_boxes)):
            newline=pd.DataFrame({'filename':sname,
                                  'score':out_scores[n],
                                  'top':out_boxes[n,0],
                                  'left':out_boxes[n,1],
                                  'bottom':out_boxes[n,2],
                                  'right':out_boxes[n,3],
                                  'classid':out_classes[n],
                                  'classname':out_classesname[n]},index=[key_index])
            df=df.append(newline)
            key_index+=1
        #进度条显示
        process_bar.show_process(int(100. * i / num_images),'','%d/%d'%(i,num_images)) 
    process_bar.show_process(100,'','%d/%d'%(num_images,num_images))
    print(df)
    #保存结果到CSV文件
    df.to_csv(results_file)
    #关闭会话
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--annopath', type=str, default='%s/data/VOCdevkit/VOC2012/Annotations'%(os.getenv('HOME')),
        help='Annotations path'
    )

    parser.add_argument(
        '--JPEGImages', type=str, default='%s/data/VOCdevkit/VOC2012/JPEGImages'%(os.getenv('HOME')),
        help='Jpeg images path'
    )

    parser.add_argument(
        '--imagesetfile', type=str, default='%s/data/VOCdevkit/VOC2012/ImageSets/Main/val.txt'%(os.getenv('HOME')),
        help='test iamges set file'
    )

    parser.add_argument(
        '--resultfile', type=str, default='%s/data/VOCdevkit/VOC2012/results/result.txt'%(os.getenv('HOME')),
        help='results file'
    )

    FLAGS = parser.parse_args()

    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    if "input" in FLAGS:
        print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
    detect_img(FLAGS,YOLO(**vars(FLAGS)))  #vars() 函数返回对象object的属性和属性值的字典对象


