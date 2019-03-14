#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''numpy重要操作汇总
numpy 基础入门 - 30分钟学会numpy
    https://m.pythontab.com/article/1271
'''

'''
Numpy简单介绍
1.Numpy是什么
很简单，Numpy是Python的一个科学计算的库，提供了矩阵运算的功能，其一般与Scipy、matplotlib一起使用。其实，list已经提供了类似于矩阵的表示形式，不过numpy为我们提供了更多的函数。如果接触过matlab、scilab，那么numpy很好入手。 在以下的代码示例中，总是先导入了numpy：（通用做法import numpu as np 简单输入）
'''
import numpy as np  
import numpy.linalg as nplg
print("numpy version:",np.__version__)

'''2. 多维数组
多维数组的类型是：numpy.ndarray。
使用numpy.array方法
'''
#以list或tuple变量为参数产生一维数组：
print(np.array([1,2,3,4]))
#[1 2 3 4]  
print(np.array((1.2,2,3,4)))
#[ 1.2  2.   3.   4. ]  
print(type(np.array((1.2,2,3,4))))
#<type 'numpy.ndarray'>

#以list或tuple变量为元素产生二维数组或者多维数组：
x = np.array(((1,2,3),(4,5,6)))  
print(x)  
#array([[1, 2, 3],  
#       [4, 5, 6]])  
y = np.array([[1,2,3],[4,5,6]])  
print(y)  
#array([[1, 2, 3],  
#       [4, 5, 6]])

'''numpy数据类型设定与转换
    numpy ndarray数据类型可以通过参数dtype 设定，而且可以使用astype转换类型，在处理文件时候这个会很实用，注意astype 调用会返回一个新的数组，也就是原始数据的一份拷贝。
'''
numeric_strings2 = np.array(['1.23','2.34','3.45'],dtype=np.string_)  #注意：字符串类型 np.string_
print(numeric_strings2)
#array(['1.23', '2.34', '3.45'],   
#      dtype='|S4')  
print(numeric_strings2.astype(float))
#array([ 1.23,  2.34,  3.45])

#numpy索引与切片
#index 和slicing ：第一数值类似数组横坐标，第二个为纵坐标
x = np.array(((1,2,3),(4,5,6)))  
print(x[1,2]  )
#6  
y=x[:,1]  
print(y)  
#array([2, 5])

#涉及改变相关问题，我们改变上面y是否会改变x？这是特别需要关注的！
print(y)  
#array([2, 5])  
y[0] = 10  
print(y)  
#array([10,  5])  
print(x)  
#array([[ 1, 10,  3],  
#       [ 4,  5,  6]])
#通过上面可以发现改变y会改变x ，因而我们可以推断，y和x指向是同一块内存空间值，系统没有为y 新开辟空间把x值赋值过去。

arr = np.arange(10)  
print(arr)  
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  
print(arr[4])
#4  
print(arr[3:6]  )
#array([3, 4, 5])  
arr[3:6] = 12  
print(arr)  
#array([ 0,  1,  2, 12, 12, 12,  6,  7,  8,  9])
#如上所示：当将一个标量赋值给切片时，该值会自动传播整个切片区域，这个跟列表最重要本质区别，数组切片是原始数组的视图，视图上任何修改直接反映到源数据上面。
#思考为什么这么设计？ Numpy 设计是为了处理大数据，如果切片采用数据复制话会产生极大的性能和内存消耗问题。


#假如说需要对数组是一份副本而不是视图可以如下操作：
arr_copy = arr[3:6].copy()  
arr_copy[:]=24  
print(arr_copy)
#array([24, 24, 24])  
print(arr)  
#array([ 0,  1,  2, 12, 12, 12,  6,  7,  8,  9])

#再看下对list 切片修改
l=range(10)  
print(l)
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
'''
l[5:8] = 12  
Traceback (most recent call last):  
   
  File "<ipython-input-36-022af3ddcc9b>", line 1, in <module>  
    l[5:8] = 12  
   
TypeError: can only assign an iterable  
'''
l1= l[5:8]  
print(l1)  
#[5, 6, 7]  
#l1[0]=12  
#Traceback (most recent call last):
#  File "demo_np.py", line 107, in <module>
#    l1[0]=12  
#TypeError: 'range' object does not support item assignment
print(l1)  
#[12, 6, 7]  
print(l)  
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#这里设计到python 中深浅拷贝，其中切片属于浅拷贝


#多维数组索引、切片
arr2d = np.arange(1,10).reshape(3,3)  
print(arr2d)  
#array([[1, 2, 3],  
#       [4, 5, 6],  
#       [7, 8, 9]])  
print(arr2d[2])
#array([7, 8, 9])  
print(arr2d[0][2])
#3  

#布尔型索引
#这种类型在实际代码中出现比较多，关注下。
names = np.array(['Bob','joe','Bob','will'])  
print(names == 'Bob')
#array([ True, False,  True, False], dtype=bool)
data=np.array([[ 0.36762706, -1.55668952,  0.84316735, -0.116842  ],  
       [ 1.34023966,  1.12766186,  1.12507441, -0.68689309],  
       [ 1.27392366, -0.43399617, -0.80444728,  1.60731881],  
       [ 0.23361565,  1.38772715,  0.69129479, -1.19228023],  
       [ 0.51353082,  0.17696698, -0.06753478,  0.80448168],  
       [ 0.21773096,  0.60582802, -0.46446071,  0.83131122],  
       [ 0.50569072,  0.04431685, -0.69358155, -0.9629124 ]])  
data[data < 0] = 0  
print(data)
#array([[ 0.36762706,  0.        ,  0.84316735,  0.        ],  
#       [ 1.34023966,  1.12766186,  1.12507441,  0.        ],  
#       [ 1.27392366,  0.        ,  0.        ,  1.60731881],  
#       [ 0.23361565,  1.38772715,  0.69129479,  0.        ],  
#       [ 0.51353082,  0.17696698,  0.        ,  0.80448168],  
#       [ 0.21773096,  0.60582802,  0.        ,  0.83131122],  
#       [ 0.50569072,  0.04431685,  0.        ,  0.        ]])
#上面展示通过布尔值来设置值的手段。


#数组文件输入输出
#在跑实验时经常需要用到读取文件中的数据，其实在numpy中已经有成熟函数封装好了可以使用
#将数组以二进制形式格式保存到磁盘，np.save 、np.load 函数是读写磁盘的两个主要函数，默认情况下，数组以未压缩的原始二进制格式保存在扩展名为.npy的文件中
arr = np.arange(10)  
np.save('some_array',arr)  
np.load('some_array.npy')  
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


#存取文本文件：
#文本中存放是聚类需要数据，直接可以方便读取到numpy array中，省去一行行读文件繁琐。
#arr = np.loadtxt('dataMatrix.txt',delimiter=' ')  
#print(arr)  
#array([[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,  
#         0.8125    ],  
#       [ 0.52882353,  0.56271186,  0.48220588,  0.53384615,  0.61651376,  
#         0.58285714],  
#       [ 0.        ,  0.        ,  0.        ,  1.        ,  1.        ,  
#         1.        ],  
#       [ 1.        ,  0.92857143,  0.91857143,  1.        ,  1.        ,  
#         1.        ],  
#       [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,  
#         1.        ],  
#       [ 0.05285714,  0.10304348,  0.068     ,  0.06512821,  0.05492308,  
#         0.05244898],  
#       [ 0.04803279,  0.08203125,  0.05516667,  0.05517241,  0.04953488,  
#         0.05591549],  
#       [ 0.04803279,  0.08203125,  0.05516667,  0.05517241,  0.04953488,  
#         0.05591549]])
#np.savetxt 执行相反的操作，这两个函数在跑实验加载数据时可以提供很多便利！！！


#使用numpy.arange方法
print(np.arange(15)  )
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]  
print(type(np.arange(15)))
#<type 'numpy.ndarray'>  
print(np.arange(15).reshape(3,5))
#[[ 0  1  2  3  4]  
# [ 5  6  7  8  9]  
# [10 11 12 13 14]]  
print(type(np.arange(15).reshape(3,5)))
#<type 'numpy.ndarray'>

#使用numpy.linspace方法
#例如，在从1到10中产生20个数：
print(np.linspace(1,10,20)) 
#[  1.           1.47368421   1.94736842   2.42105263   2.89473684  
#   3.36842105   3.84210526   4.31578947   4.78947368   5.26315789  
#   5.73684211   6.21052632   6.68421053   7.15789474   7.63157895  
#   8.10526316   8.57894737   9.05263158   9.52631579  10.        ]


#使用numpy.zeros，numpy.ones，numpy.eye等方法可以构造特定的矩阵
print(np.zeros((3,4)))
#[[ 0.  0.  0.  0.]  
# [ 0.  0.  0.  0.]  
# [ 0.  0.  0.  0.]]  
print(np.ones((3,4)))
#[[ 1.  1.  1.  1.]  
# [ 1.  1.  1.  1.]  
# [ 1.  1.  1.  1.]]  
print(np.eye(3))
#[[ 1.  0.  0.]  
# [ 0.  1.  0.]  
# [ 0.  0.  1.]]


#获取数组的属性
a = np.zeros((2,2,2))  
print(a.ndim)   #数组的维数  
#3  
print(a.shape)  #数组每一维的大小  
#(2, 2, 2)  
print(a.size)   #数组的元素数  
#8
print(a.dtype)  #元素类型  >";
#float64  
print(a.itemsize)  #每个元素所占的字节数  
#8


#合并数组
#使用numpy下的vstack（垂直方向）和hstack（水平方向）函数：
a = np.ones((2,2))  
b = np.eye(2)  
print(np.vstack((a,b)))
#[[ 1.  1.]  
# [ 1.  1.]  
# [ 1.  0.]  
# [ 0.  1.]]  
print(np.hstack((a,b)))
#[[ 1.  1.  1.  0.]  
# [ 1.  1.  0.  1.]]

#看一下这两个函数有没有涉及到浅拷贝这种问题：
c = np.hstack((a,b))  
print(c)  
#[[ 1.  1.  1.  0.]  
# [ 1.  1.  0.  1.]]  
a[1,1] = 5  
b[1,1] = 5
print(c)  
#[[ 1.  1.  1.  0.]  
# [ 1.  1.  0.  1.]]
#通过上面可以知道，这里进行是深拷贝，而不是引用指向同一位置的浅拷贝。

#深拷贝数组
#数组对象自带了浅拷贝和深拷贝的方法，但是一般用深拷贝多一些：
a = np.ones((2,2))  
b = a         #浅拷贝  
print(b is a) 
#True  
c = a.copy()  #深拷贝  
print(c is a) 
#False


#基本的矩阵运算
#转置：
a = np.array([[1,0],[2,3]])  
print(a)  
#[[1 0]  
# [2 3]]  
print(a.transpose())
#[[1 2]  
# [0 3]]

#numpy.linalg模块中有很多关于矩阵运算的方法：
#特征值、特征向量：
a = np.array([[1,0],[2,3]])  
print(nplg.eig(a)) 
#(array([ 3.,  1.]), array([[ 0.        ,  0.70710678],  
#       [ 1.        , -0.70710678]]))


#numpy.random模块中提供啦大量的随机数相关的函数。
#1 numpy中产生随机数的方法
#　　1)rand() 　　产生[0,1]的浮点随机数,括号里面的参数可以指定产生数组的形状
#　　2)randn()　　产生标准正太分布随机数，参数含义与random相同
#　　3)randint()　　产生指定范围的随机数，最后一个参数是元祖，他确定数组的形状
 
#只显示小数点后两位
np.set_printoptions(precision = 2)
r1 = np.random.rand(3,4)                    #3x4均匀分布
r2 = np.random.randn(5,4)                   #5x4正态分布
r3 = np.random.randint(0,10,size = (4,3))   #0-10随机整数，shape=(4,3)
print(r1)
print(r2)
print(r3)
'''
[[ 0.34  0.51  0.65  0.57]
 [ 0.97  0.16  0.62  0.37]
 [ 0.23  0.78  0.77  0.46]]
[[-0.69 -1.24 -0.32  1.07]
 [ 0.05 -1.97  1.01 -1.59]
 [ 1.51 -1.21  1.02 -0.19]
 [ 1.49 -0.42  0.64  0.07]
 [-0.1   1.11  0.24 -0.18]]
[[9 6 7]
 [1 9 7]
 [4 9 6]
 [3 9 0]]
'''


# 2 常用分布
#　　1）normal(）　正太分布
#　　2）uniform()　　均匀分布
#　　3）poisson()　　泊松分布
#第一个参数是均值，第二个参数是标准差
r1 = np.random.normal(100,10,size = (3,4))
print(r1)
 
#前两个参数分别是区间的初始值和终值
r2 = np.random.uniform(0,10,size = (3,4))
print(r2)
 
#第一个参数为指定的lanbda系数
r3 = np.random.poisson(2.0,size = (3,4))
print(r3)
'''
[[ 100.67   98.39   99.36  103.37]
 [  98.23   95.11  107.57  111.23]
 [  97.26   75.21  110.4   112.53]]
[[ 2.42  6.81  9.96  3.15]
 [ 9.28  4.4   7.87  5.19]
 [ 3.47  2.92  4.5   2.58]]
[[3 1 5 0]
 [1 0 4 3]
 [3 1 2 1]]
'''

#3 乱序和随机抽取
#　　permutation()随机生成一个乱序数组，当参数是n时，返回[0,n)的乱序，他返回一个新数组。而shuffle()则直接将原数组打乱。choice（）是从指定的样本中随机抽取。
#返回打乱数组，原数组不变
r1 = np.random.randint(10,100,size = (3,4))
print(r1)
print(np.random.permutation(r1))
print(r1) 
print(np.random.permutation(5))
 
# 使用shuffle打乱数组顺序
x = np.arange(10)
np.random.shuffle(x)
print(x)
 
#choice()函数从指定数组中随机抽取样本
#size参数用于指定输出数组的大小
#replace参数为True时，进行可重复抽取，而False表示进行不可重复的抽取。默认为True
x = np.array(10)
c1 = np.random.choice(x,size = (2,3))
print(c1)
 
c2 = np.random.choice(x,5,replace = False)
print (c2)
'''
[[78 22 43 70]
 [46 87 12 32]
 [11 56 89 79]]
[[11 56 89 79]
 [78 22 43 70]
 [46 87 12 32]]
[[78 22 43 70]
 [46 87 12 32]
 [11 56 89 79]]
[4 1 2 0 3]
[3 4 9 5 8 2 7 0 6 1]
[[4 7 9]
 [9 1 7]]
[5 3 2 8 4]
'''

