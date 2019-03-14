#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Tensorflow数据运算
'''
#import pdb
#pdb.set_trace()
import tensorflow as tf
import numpy as np

sess=tf.InteractiveSession() #创建会话
#=================TensorFLow 数学运算===================
'''一、Tensor 之间的运算规则
    相同大小 Tensor 之间的任何算术运算都会将运算应用到元素级
    不同大小 Tensor(要求dimension 0 必须相同) 之间的运算叫做广播(broadcasting)
    Tensor 与 Scalar(0维 tensor) 间的算术运算会将那个标量值传播到各个元素
    Note： TensorFLow 在进行数学运算时，一定要求各个 Tensor 数据类型一致
'''
#二、常用操作符和基本数学函数
#    大多数运算符都进行了重载操作，使我们可以快速使用 (+ - * /) 等，但是有一点不好的是使用重载操作符后就不能为每个操作命名了。
#-------------------------
# 算术操作符：+ - * / % 
#tf.add(x, y, name=None)        # 加法(支持 broadcasting)
x1=tf.constant([1,2,3])
x2=tf.constant([2,3,4])
y=x1+x2
print(y.eval()) #=>[3,5,7]
#tf.subtract(x, y, name=None)   # 减法
x1=tf.constant([1,2,3])
x2=tf.constant([2,3,4])
y=x1-x2
print(y.eval()) #=>[-1,-1,-1]
#tf.multiply(x, y, name=None)   # 乘法
x1=tf.constant([1,2,3])
x2=tf.constant([2,3,4])
y=tf.multiply(x1,x2)
print(y.eval()) #=>[2,6,12]
#tf.divide(x, y, name=None)     # 浮点除法, 返回浮点数(python3 除法)
x1=tf.constant([1,2,3])
x2=tf.constant([2,3,4])
y=tf.divide(x1,x2)
print(y.eval()) #=>[0.5        0.66666667 0.75      ]
#tf.mod(x, y, name=None)        # 取余
x1=tf.constant([5,2,3])
x2=tf.constant([2,3,4])
y=tf.mod(x1,x2)
print(y.eval()) #=>[1,2,3]

#-------------------------
# 幂指对数操作符：^ ^2 ^0.5 e^ ln 
#tf.pow(x, y, name=None)        # 幂次方
x1=tf.constant([5,2,3])
x2=tf.constant(2)
y=tf.pow(x1,x2) 
print(y.eval()) #=>[25,4,9]
x1=tf.constant([5,2,3])
x2=tf.constant([1,2,3])
y=tf.pow(x1,x2)
print(y.eval()) #=>[5,4,27]
#tf.square(x, name=None)        # 平方
x=tf.constant([1,2,3])
y=tf.square(x) 
print(y.eval()) #=>[1,4,9]
#tf.sqrt(x, name=None)          # 开根号，必须传入浮点数或复数
x=tf.constant([1,4,9],dtype=tf.float32)
y=tf.sqrt(x)
print(y.eval()) #=>[1,2,3]
#tf.exp(x, name=None)           # 计算 e 的次方
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.exp(x)
print(y.eval()) #=>[e^1,e^2,e^3]
#tf.log(x, name=None)           # 以 e 为底，必须传入浮点数或复数
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.log(x)
print(y.eval()) #=>[log(1),log(2),log(3)]


#---------------------------------
# 取符号、负、倒数、绝对值、近似、两数中较大/小的
#tf.negative(x, name=None)      # 取负(y = -x).
x=tf.constant([1,2,-3])
y=tf.negative(x)
print(y.eval()) #=>[-1,-2,3]
#tf.sign(x, name=None)          # 返回 x 的符号
x=tf.constant([2,0,-2])
y=tf.sign(x)
print(y.eval()) #=>[1,0,-1]
#tf.reciprocal(x, name=None)    # 取倒数
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.reciprocal(x)
print(y.eval()) #=[1/1,1/2,1/3]
#tf.abs(x, name=None)           # 求绝对值
x=tf.constant([1,2,-3])
y=tf.abs(x)
print(y.eval()) #=>[1,2,3]
#tf.round(x, name=None)         # 四舍五入
x=tf.constant([1.3,2.6,3.7])
y=tf.round(x)
print(y.eval()) #=>[1,3,4]
#tf.ceil(x, name=None)          # 向上取整
x=tf.constant([1.3,2.6,3.7])
y=tf.ceil(x)
print(y.eval()) #=>[2,3,4]
#tf.floor(x, name=None)         # 向下取整
x=tf.constant([1.3,2.6,3.7])
y=tf.floor(x)
print(y.eval()) #=>[1,2,3]
#tf.rint(x, name=None)          # 取最接近的整数 
x=tf.constant([-1.2,1.4,1.6])
y=tf.rint(x)
print(y.eval()) #=>[-1,1,2]
#tf.maximum(x, y, name=None)    # 返回两tensor中的最大值 (x > y ? x : y)
x1=tf.constant([1,2,3],dtype=tf.float32)
x2=tf.constant([3,1,5],dtype=tf.float32)
y=tf.maximum(x1,x2)
print(y.eval()) #=>[3,2,5]
tf.minimum(x, y, name=None)    # 返回两tensor中的最小值 (x < y ? x : y)
x1=tf.constant([1,2,3],dtype=tf.float32)
x2=tf.constant([3,1,5],dtype=tf.float32)
y=tf.maximum(x1,x2)
print(y.eval()) #=>[1,1,3]


#---------------------------
# 三角函数和反三角函数
#tf.cos(x, name=None)    
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.cos(x)
print(y.eval()) #=>[cos(1),cos(2),cos(3)]
#tf.sin(x, name=None)
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.sin(x)
print(y.eval()) #=>[sin(1),sin(2),sin(3)]
#tf.tan(x, name=None)    
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.tan(x)
print(y.eval()) #=>[tan(1),tan(2),tan(3)]
#tf.acos(x, name=None)
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.acos(x)
print(y.eval()) #=>[acos(1),acos(2),acos(3)]
#tf.asin(x, name=None)
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.asin(x)
print(y.eval()) #=>[asin(1),asin(2),asin(3)]
#tf.atan(x, name=None)   
x=tf.constant([1,2,3],dtype=tf.float32)
y=tf.atan(x)
print(y.eval()) #=>[atan(1),atan(2),atan(3)]


#---------------------------------
# 其它
#tf.cast(x,DType,name=None) #转换数据类型
x=tf.constant([1,2,3],tf.float32)
y=tf.cast(x,tf.int32)
#tf.div(x, y, name=None)  # python 2.7 除法, x/y-->int or x/float(y)-->float
x1=tf.constant([11,3,14])
x2=tf.constant([2,3,4])
y1=tf.div(x1,x2)
y2=tf.div(tf.cast(x1,tf.float32),tf.cast(x2,tf.float32))
print(y1.eval()) #=>[5,1,3]
print(y2.eval()) #=>[5.5,1,3.5]
#tf.truediv(x, y, name=None) # python 3 除法, x/y-->float
x1=tf.constant([11,3,14])
x2=tf.constant([2,3,4])
y1=tf.truediv(x1,x2)
y2=x1/x2
print(y1.eval()) #=>[5.5,1,3.5]
print(y2.eval()) #=>[5.5,1,3.5]
#tf.floordiv(x, y, name=None)  # python 3 除法, x//y-->int
x1=tf.constant([11,3,14])
x2=tf.constant([2,3,4])
y1=tf.floordiv(x1,x2)
y2=x1//x2
print(y1.eval()) #=>[5,1,3] 
print(y2.eval()) #=>[5,1,3]
#tf.realdiv(x, y, name=None)  #实数除，参数必须为浮点数类型
x1=tf.constant([1,2,3],tf.float32)
x2=tf.constant([2,4,10],tf.float32)
y=tf.realdiv(x1,x2)
print(y.eval()) #=>[0.5,0.5,0.3]
#tf.truncatediv(x, y, name=None) ?????
#tf.floor_div(x, y, name=None) #地板除，同 x//y
#tf.truncatemod(x, y, name=None) ?????
#tf.floormod(x, y, name=None)    ?????
#tf.cross(x, y, name=None)       ?????
#tf.add_n(inputs, name=None)  # inputs: A list of Tensor objects, each with same shape and type   ?????
#tf.squared_difference(x, y, name=None)  ??????


#三、矩阵数学函数
# 矩阵乘法(tensors of rank >= 2)
#tf.matmul(a, b, transpose_a=False, transpose_b=False,    adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
x1=tf.constant([[1,2,3],[4,5,6]])
x2=tf.constant([[1,2],[3,4],[5,6]])
y=tf.matmul(x1,x2)
print(y.eval())

# 转置，可以通过指定 perm=[1, 0] 来进行轴变换
#tf.transpose(a, perm=None, name='transpose')
x=tf.constant([[1,2,3],[4,5,6]])
y=tf.transpose(x)

# 在张量 a 的最后两个维度上进行转置
#tf.matrix_transpose(a, name='matrix_transpose')
# Matrix with two batch dimensions, x.shape is [1, 2, 3, 4]
# tf.matrix_transpose(x) is shape [1, 2, 4, 3]
x=np.random.rand(2,3,4)
y=tf.matrix_transpose(x)

# 求矩阵的迹
#tf.trace(x, name=None)
x=tf.constant([[1,2,3],[4,5,6],[7,8,9]])
y=tf.trace(x)

# 计算方阵行列式的值
#tf.matrix_determinant(input, name=None)
x=tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
y=tf.matrix_determinant(x)

# 求解可逆方阵的逆，input 必须为浮点型或复数
#tf.matrix_inverse(input, adjoint=None, name=None)
x=tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
y=tf.matrix_inverse(x)

# 奇异值分解
#tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)
x=tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
y=tf.svd(x)

# QR 分解
#tf.qr(input, full_matrices=None, name=None)
x=tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
y=tf.svd(x)

# 求张量的范数(默认2)
#tf.norm(tensor, ord='euclidean', axis=None, keep_dims=False, name=None)
x=tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32)
y=tf.norm(x)


# 构建一个单位矩阵, 或者 batch 个矩阵，batch_shape 以 list 的形式传入
#tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32, name=None)
# Construct one identity matrix.
tf.eye(2)
#==> [[1., 0.],
#     [0., 1.]]

# Construct a batch of 3 identity matricies, each 2 x 2.
# batch_identity[i, :, :] is a 2 x 2 identity matrix, i = 0, 1, 2.
batch_identity = tf.eye(2, batch_shape=[3])

# Construct one 2 x 3 "identity" matrix
tf.eye(2, num_columns=3)
#==> [[ 1.,  0.,  0.],
#     [ 0.,  1.,  0.]]


# 构建一个对角矩阵，rank = 2*rank(diagonal)
#tf.diag(diagonal, name=None)
# 'diagonal' is [1, 2, 3, 4]
tf.diag([1,2,3])
# ==> [[1, 0, 0, 0]
#      [0, 2, 0, 0]
#      [0, 0, 3, 0]
#      [0, 0, 0, 4]]

# 其它
#tf.diag_part
#tf.matrix_diag
#tf.matrix_diag_part
#tf.matrix_band_part
#tf.matrix_set_diag
#tf.cholesky
#tf.cholesky_solve
#tf.matrix_solve
#tf.matrix_triangular_solve
#tf.matrix_solve_ls
#tf.self_adjoint_eig
#tf.self_adjoint_eigvals


#四、Reduction：reduce various dimensions of a tensor
# 计算输入 tensor 所有元素的和，或者计算指定的轴所有元素的和
#tf.reduce_sum(input_tensor, axis=None, keep_dims=False, name=None)
# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
x=np.ones([2,3])
tf.reduce_sum(x)    #==> 6
tf.reduce_sum(x, 0) #==> [2, 2, 2]
tf.reduce_sum(x, 1) #==> [3, 3]
tf.reduce_sum(x, 1, keepdims=True) #==> [[3], [3]]  # 维度不缩减
tf.reduce_sum(x, [0, 1]) #==> 6


# 计算输入 tensor 所有元素的均值/最大值/最小值/积/逻辑与/或
# 或者计算指定的轴所有元素的均值/最大值/最小值/积/逻辑与/或(just like reduce_sum)
#tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None)
#tf.reduce_max(input_tensor, axis=None, keep_dims=False, name=None)
#tf.reduce_min(input_tensor, axis=None, keep_dims=False, name=None)
#tf.reduce_prod(input_tensor, axis=None, keep_dims=False, name=None)
#tf.reduce_all(input_tensor, axis=None, keep_dims=False, name=None)  # 全部满足条件
#tf.reduce_any(input_tensor, axis=None, keep_dims=False, name=None) #至少有一个满足条件


#-------------------------------------------
# 分界线以上和 Numpy 中相应的用法完全一致
#-------------------------------------------
# inputs 为一 list, 计算 list 中所有元素的累计和，
# tf.add(x, y， name=None)只能计算两个元素的和，此函数相当于扩展了其功能
#tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)


# Computes log(sum(exp(elements across dimensions of a tensor)))
#tf.reduce_logsumexp(input_tensor, axis=None, keep_dims=False, name=None)


# Computes number of nonzero elements across dimensions of a tensor
#tf.count_nonzero(input_tensor, axis=None, keep_dims=False, name=None)

#五、Scan：perform scans (running totals) across one axis of a tensor
# Compute the cumulative sum of the tensor x along axis
#tf.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)
# Eg:
#tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
#tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
#tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
#tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]


# Compute the cumulative product of the tensor x along axis
#tf.cumprod(x, axis=0, exclusive=False, reverse=False, name=None)

#六、Segmentation
#    沿着第一维(x 轴)根据 segment_ids(list)分割好相应的数据后再进行操作
#这里写图片描述

# Computes the sum/mean/max/min/prod along segments of a tensor
#tf.segment_sum(data, segment_ids, name=None)
# Eg:
m = tf.constant([5,1,7,2,3,4,1,3])
s_id = [0,0,0,1,2,2,3,3]
sess.run(tf.segment_sum(m, segment_ids=s_id))
#>array([13,  2,  7,  4], dtype=int32)

#tf.segment_mean(data, segment_ids, name=None)
#tf.segment_max(data, segment_ids, name=None)
#tf.segment_min(data, segment_ids, name=None)
#tf.segment_prod(data, segment_ids, name=None)


# 其它
#tf.unsorted_segment_sum
#tf.sparse_segment_sum
#tf.sparse_segment_mean
#tf.sparse_segment_sqrt_n

#七、 序列比较与索引提取
# 比较两个 list 或者 string 的不同，并返回不同的值和索引
#tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)

# 返回 x 中的唯一值所组成的tensor 和原 tensor 中元素在现 tensor 中的索引
#tf.unique(x, out_idx=None, name=None)

# x if condition else y, condition 为 bool 类型的，可用tf.equal()等来表示
# x 和 y 的形状和数据类型必须一致
#tf.where(condition, x=None, y=None, name=None)

# 返回沿着坐标轴方向的最大/最小值的索引
#tf.argmax(input, axis=None, name=None, output_type=tf.int64)
#tf.argmin(input, axis=None, name=None, output_type=tf.int64)


# x 的值当作 y 的索引，range(len(x)) 索引当作 y 的值
# y[x[i]] = i for i in [0, 1, ..., len(x) - 1]
#tf.invert_permutation(x, name=None)


# 其它
#tf.edit_distance


