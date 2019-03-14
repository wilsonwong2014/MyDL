#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''tf.Tensor及衍生类型汇总
    Tensor---张量，为计算图提供数据存储范式，本身不存储数据值，但可以通过session计算
    tf.Tensor
    tf.Constant ---特殊形式张量
    tf.Varianet ---特殊形式张量

Properties
    device
        The name of the device on which this tensor will be produced, or None.
    dtype
        The DType of elements in this tensor.
    graph
        The Graph that contains this tensor.
    name
        The string name of this tensor.
    op
        The Operation that produces this tensor as an output.
    shape
        Returns the TensorShape that represents the shape of this tensor.

tf.random_normal(shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
    )

tf.constant
    tf.constant(
        value,
        dtype=None,
        shape=None,
        name='Const',
        verify_shape=False
    )
Args:
    value: A constant value (or list) of output type dtype.
    dtype: The type of the elements of the resulting tensor.
    shape: Optional dimensions of resulting tensor.
    name: Optional name for the tensor.
    verify_shape: Boolean that enables verification of a shape of values.


tf.Variable
__init__(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None,
    constraint=None
)
Args:

    initial_value: A Tensor, or Python object convertible to a Tensor, which is the initial value for the Variable. 
        The initial value must have a shape specified unless validate_shape is set to False. 
        Can also be a callable with no argument that returns the initial value when called. 
        In that case, dtype must be specified. (Note that initializer functions from init_ops.py must first be bound to a shape before being used here.)
    trainable: If True, the default, also adds the variable to the graph collection GraphKeys.TRAINABLE_VARIABLES. 
        This collection is used as the default list of variables to use by the Optimizer classes.
    collections: List of graph collections keys. The new variable is added to these collections. Defaults to [GraphKeys.GLOBAL_VARIABLES].
    validate_shape: If False, allows the variable to be initialized with a value of unknown shape. If True, the default, the shape of initial_value must be known.
    caching_device: Optional device string describing where the Variable should be cached for reading. 
        Defaults to the Variable's device. If not None, caches on another device. 
        Typical use is to cache on the device where the Ops using the Variable reside, to deduplicate copying through Switch and other conditional statements.
    name: Optional name for the variable. Defaults to 'Variable' and gets uniquified automatically.
    variable_def: VariableDef protocol buffer. 
        If not None, recreates the Variable object with its contents, referencing the variable's nodes in the graph, which must already exist. The graph is not changed. 
        variable_def and the other arguments are mutually exclusive.
    dtype: If set, initial_value will be converted to the given type. 
        If None, either the datatype will be kept (if initial_value is a Tensor), or convert_to_tensor will decide.
    expected_shape: A TensorShape. If set, initial_value is expected to have this shape.
    import_scope: Optional string. Name scope to add to the Variable. Only used when initializing from protocol buffer.
    constraint: An optional projection function to be applied to the variable after being updated by an Optimizer (e.g. used to implement norm constraints or value constraints for layer weights). 
        The function must take as input the unprojected Tensor representing the value of the variable and return the Tensor for the projected value (which must have the same shape). 
        Constraints are not safe to use when doing asynchronous distributed training.

tf.placeholder(
    dtype,
    shape=None,
    name=None
)
Args:
    dtype: The type of elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any shape.
    name: A name for the operation (optional).

'''
#import pdb
#pdb.set_trace()
import tensorflow as tf
import numpy as np
#常量 tf.Constant
sess=tf.InteractiveSession() #创建会话

#============constent===============
#constant命名规则，通过关键参数设置 name='name',默认名称：const；相同名称规则：name_序号
#const1.name => const1:0  ;冒号左边是名称，右边是输出张量序号

const1=tf.constant(1,name='const1')
print(const1.name) 
#=> const1:0
print(type(const1))
#=><class 'tensorflow.python.framework.ops.Tensor'>
print(const1)
#=>Tensor("const1:0", shape=(), dtype=int32)
print(const1.eval())
#=>1
#打印属性
print(const1.device)
#=>空
print(const1.dtype)
#=><dtype: 'int32'>
print(const1.graph)
#=><tensorflow.python.framework.ops.Graph object at 0x7f50fae79208>
print(const1.name)
#=>const1:0
print(const1.op)
#=>
#name: "const1"
#op: "Const"
#attr {
#  key: "dtype"
#  value {
#    type: DT_INT32
#  }
#}
#attr {
#  key: "value"
#  value {
#    tensor {
#      dtype: DT_INT32
#      tensor_shape {
#      }
#      int_val: 1
#    }
#  }
#}
print(const1.shape)
#=>()


const2=tf.constant([[1,2,3],[4,5,6]],tf.float32,name='const1')
print(const2.name)
#=>const1_1:0
print(type(const2))
#=><class 'tensorflow.python.framework.ops.Tensor'>
print(const2)
#=>Tensor("const1_1:0", shape=(2, 3), dtype=float32)
print(const2.eval())
#[[1. 2. 3.]
#[4. 5. 6.]]

const3=tf.constant(1,shape=(2,3))
print(const3)
#=>Tensor("Const:0", shape=(2, 3), dtype=int32)
print(const3.eval())
#[[1 1 1]
# [1 1 1]]

const4=tf.constant([1,2,3,4],shape=(2,3)) #当初始化值与给定shape不匹配时,缺少部分以最后一个元素填充
print(const4)
#=>Tensor("Const_1:0", shape=(2, 3), dtype=int32)
print(const4.eval())
#[[1 2 3]
# [4 4 4]]


#============Variant===============
v1=tf.Variable(np.random.rand(2,3),name='v1') #声明变量
v2=tf.get_variable("v2", [2,3], initializer = tf.zeros_initializer()) # 设置初始值为0
v3=tf.get_variable("v3", [2,3], initializer = tf.ones_initializer()) # 设置初始值为1
#变量初始化
sess.run(v1.initializer) #或 v1.initializer.run()
sess.run(v2.initializer) #或 v2.initializer.run()
sess.run(v3.initializer) #或 v3.initializer.run()
#初始化所有变量
tf.global_variables_initializer().run()


#==========placeholder=============
x=tf.placeholder(tf.float32, shape=(None,2,3), name=None)
#    dtype：表示tensorflow中的数据类型，如常用的tf.float32,tf.float64等数值类型;
#    shape：表示数据类型，默认的None是一个一维的数值，shape=[None,5],表示行不定，列是5;
#    name：张量名称;


