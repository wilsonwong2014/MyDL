#!/sur/bin/env python3
# -*- coding: utf-8 -*-

'''图像预处理ImageDataGenerator
    https://keras-cn.readthedocs.io/en/latest/preprocessing/image/

    图片生成器ImageDataGenerator

    keras.preprocessing.image.ImageDataGenerator
       (featurewise_center           =False,
        samplewise_center            =False,
        featurewise_std_normalization=False,
        samplewise_std_normalization =False,
        zca_whitening                =False,
        zca_epsilon                  =1e-6,
        rotation_range               =0.,
        width_shift_range            =0.,
        height_shift_range           =0.,
        shear_range                  =0.,
        zoom_range                   =0.,
        channel_shift_range          =0.,
        fill_mode                    ='nearest',
        cval                         =0.,
        horizontal_flip              =False,
        vertical_flip                =False,
        rescale                      =None,
        preprocessing_function       =None,
        data_format                  =K.image_data_format()
       )
    用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。
    参数
        featurewise_center:           布尔值，使输入数据集去中心化（均值为0）, 按feature执行
        samplewise_center:            布尔值，使输入数据的每个样本均值为0
        featurewise_std_normalization:布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization: 布尔值，将输入的每个样本除以其自身的标准差
        zca_whitening:                布尔值，对输入数据施加ZCA白化
        zca_epsilon:                  ZCA使用的eposilon，默认1e-6
        rotation_range:               整数，数据提升时图片随机转动的角度
        width_shift_range:            浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        height_shift_range:           浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        shear_range:                  浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range:                   浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        channel_shift_range:          浮点数，随机通道偏移的幅度
        fill_mode:                    ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
        cval:                         浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        horizontal_flip:              布尔值，进行随机水平翻转
        vertical_flip:                布尔值，进行随机竖直翻转
        rescale:                      重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        preprocessing_function:       将被应用于每个输入的函数。该函数将在图片缩放和数据提升之后运行。
                                      该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
        data_format:                  字符串，“channel_first”或“channel_last”之一，代表图像的通道维的位置。
                                      该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。
                                      以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。
                                      该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channel_last”

    方法
        fit(x, augment=False, rounds=1): 计算依赖于数据的变换所需要的统计信息(均值方差等),只有使用featurewise_center，featurewise_std_normalization或zca_whitening时需要此函数。
            X:       numpy array，样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
            augment: 布尔值，确定是否使用随即提升过的数据
            round:   若设augment=True，确定要在数据上进行多少轮数据提升，默认值为1
            seed:    整数,随机数种子
        flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png'):
                接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
            x:           样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
            y:           标签
            batch_size:  整数，默认32
            shuffle:     布尔值，是否随机打乱数据，默认为True
            save_to_dir: None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
            save_prefix: 字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
            save_format: "png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
            yields:      形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.
            seed:        整数,随机数种子
        flow_from_directory(directory): 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
            directory:     目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用.详情请查看此脚本
            target_size:   整数tuple,默认为(256, 256). 图像将被resize成该尺寸
            color_mode:    颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
            classes:       可选参数,为子文件夹的列表,如['dogs','cats']默认为None. 若未提供,则该类别列表将从directory下的子文件夹名称/结构自动推断。
                           每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。通过属性class_indices可获得文件夹名与类的序号的对应字典。
            class_mode:    "categorical", "binary", "sparse"或None之一. 默认为"categorical. 
                           该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签.
                           "sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.
            batch_size:    batch数据的大小,默认32
            shuffle:       是否打乱数据,默认为True
            seed:          可选参数,打乱数据和进行变换时的随机数种子
            save_to_dir:   None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
            save_prefix:   字符串，保存提升后图片时使用的前缀, 仅当设置了save_to_dir时生效
            save_format:   "png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
            flollow_links: 是否访问子文件夹中的软链接
    测试数据:
        data_path="%s/data/demo/%s"%(os.getenv("HOME"),os.path.splitext(os.path.basename(sys.argv[0]))[0])
        根目录下分若干子目录，每个子目录为一类图像，如：
            ./cats/cat.1.jpg
            ./cats/cat.2.jpg
            ./dogs/dog.1.jpg
            ./dogs/dog.2.jpg

'''
import os
import sys
import keras
import pdb
pdb.set_trace() 

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras import backend as K

data_path='%s/data/demo/%s'%(os.getenv('HOME'),os.path.splitext(os.path.basename(sys.argv[0]))[0])

#构造生成器
#datagen = ImageDataGenerator(
#    featurewise_center=False,
#    samplewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True)
datagen = ImageDataGenerator(
        featurewise_center           =False,
        samplewise_center            =False,
        featurewise_std_normalization=False,
        samplewise_std_normalization =False,
        zca_whitening                =False,
        zca_epsilon                  =1e-6,
        rotation_range               =0.,
        width_shift_range            =0.,
        height_shift_range           =0.,
        shear_range                  =0.,
        zoom_range                   =0.,
        channel_shift_range          =0.,
        fill_mode                    ='nearest',
        cval                         =0.,
        horizontal_flip              =False,
        vertical_flip                =False,
        rescale                      =None,
        preprocessing_function       =None,
        data_format                  =K.image_data_format()
       )

#================加载测试数据=======================
epochs=2
num_classes=10
#测试数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#================fit 数据统计例程===================
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
#fit测试数据:不会改变源数据值，计算结果存放在datagen相关属性
x_test=x_train[0:1,:,:,:]
x_test+=10
#fit前结果:球和，均值，标准差
print('fit before:sum:%s,mean:%s,std:%s'%(x_test.sum(),datagen.mean,datagen.std))   #345694,None
datagen.fit(x_test)#只有使用featurewise_center，featurewise_std_normalization或zca_whitening时需要此函数。计算结果存放在datagen相关属性，如datagen.mean
#fit后结果:球和，均值，标准差
print('fit after:%s,%s'%(x_test.sum(),datagen.mean))

#================使用.flow()的例子==================
#把样本图像保存两个示范类别"class1,class2"
#生成数据生成器:返回tuple(narray,narray)，
flow_path='%s/flow'%(data_path)
classes=['class1','class2']
for sname in classes:
    save_to_dir='%s/%s'%(flow_path,sname) #保存目录
    os.makedirs(save_to_dir) if not os.path.exists(save_to_dir) else ''
    img_gen=datagen.flow(x_train,y_train,batch_size=5 ,save_prefix='img_',save_to_dir=save_to_dir)
    #type(img_gen1)=>'keras_preprocessing.image.NumpyArrayIterator'>
    if img_gen.n>0:
        batch=next(img_gen) #可以无限检索数据
        #type(batch1)=><class 'tuple'> ,len(batch1)=>2,第一个为数据，第二个为标签
    else:
        print('img_gen1 is Null')
#无限循环
#for x_batch,y_batch in img_gen:
#    print('x.shape:%s,y.shape:%s'%(x_batch.shape,y_batch.shape))

#=============使用.flow_from_directory的例子===============
#type(img_gen2)=><class 'keras_preprocessing.image.DirectoryIterator'>
#ImageDataGenerator对象属性：
#   img_gen2.samples ----------样本个数
#   img_gen2.classes ----------待处理子目录列表，并按顺序指定标签序号，否则由系统自动推断,例(参数):classes=['dogs','cats'];作为属性值时为子目录标签号：[1xsamples]对应img_gen2.filenames
#                              img_gen2.filenames列表对应类别号, 如:[0,0,1,1]
#   img_gen2.class_indices ----子目录标签字典，例：{'dogs':0,'cats':1}
#   img_gen2.filenames --------文件列表[所有], 如:['dog.1.jpg','dog.2.jpg','cat.1.jpg','cat.2.jpg']
flow_from_directory='%s/flow_from_directory'%(data_path)
os.makedirs(flow_from_directory) if not os.path.exists(flow_from_directory) else ''
img_gen=datagen.flow_from_directory(directory=flow_path,save_to_dir=flow_from_directory,target_size=(224,224),classes=['class1','class2'])
#type(batch2)=><class 'tuple'> ,len(batch2)=>2,第一个为数据，第二个为标签
if img_gen.n>0:
    print('img_gen.samples:%d'%(img_gen.samples))
    print('img_gen.class_indices:%s'%(img_gen.class_indices))
    print('img_gen.classes:%s'%(img_gen.classes[:40]))
    print('img_gen.filenames:%s'%(img_gen.filenames[:40]))
    batch=next(img_gen)
else:
    print('img_gen2 is Null')

#=====================datagen属性==========================
print('datagen.mean:%s'%(datagen.mean)) #均值
print('datagen.std:%s'%(datagen.std))   #标准方差
#其他查看 dir(datagen)
