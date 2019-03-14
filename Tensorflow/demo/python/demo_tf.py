#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#############################
#  TensorFlow 使用范例      #
#############################

#https://www.w3cschool.cn/doc_tensorflow_python/

###########变量############
#变量 Variable 属性
# .device
# .dtype
# .graph
# .initial_value
# .initializer
# .name
# .op
# .shape
#
#变量 Variable 方法
# assign(value,use_locking=False);
# assign_add(delta,use_locking=False);
# assign_sub(delta,use_locking=False);
# count_up_to(limit);
# eval(session=None);
# from_proto(variable_def,import_scope=None);
# get_shape();
# initialized_value();
# load(value,session=None);

###########张量############
#张量 Tensor 属性
# .device
# .dtype
# .graph
# .name
# .op
# .shape
#
#张量 Tensor方法
# consumers();
# eval(feed_dict=None,session=None);
# get_shape();
# set_shape(shape);
#
##########会话############
#会话 Session 属性
# .graph
# .graph_def
# .sess_str
#
#会话 Session 方法
# as_default();
# close();
# partial_run(handle,fetches,feed_dict=None) ;
# partial_run_setup(fetches,feeds=None);
# reset(target,containers=None,config=None);
# run(fetches,feed_dict=None,options=None,run_metadata=None);
#
##########计算图###########
#计算图Graph 属性
# .building_function
# .finalized
# .graph_def_versions
# .seed
# .version
# 
#计算图Graph 方法
# add_to_collection(name,value);
# add_to_collections(names,value);
# as_default();
# as_graph_def(from_version=None,add_shapes=False);
# as_graph_element(obj,allow_tensor=True,allow_operation=True);
# clear_collection(name);
# colocate_with(op,ignore_existing=False);
# container(container_name);
# control_dependencies(control_inputs);
# create_op(op_type,inputs,dtypes,input_types=None,name=None,attrs=None,op_def=None,compute_shapes=True,compute_device=True);
# device(device_name_or_function);
# finalize();
# get_all_collection_keys();
# get_collection(name,scope=None);
# get_collection_ref(name);
# get_name_scope();
# get_operation_by_name(name);
# get_operations();
# get_tensor_by_name(name);
# gradient_override_map(op_type_map);
# is_feedable(tensor);
# is_fetchable(tensor_or_op);
# name_scope(name);
# prevent_feeding(tensor);
# prevent_fetching(op);
# unique_name(name,mark_as_used=True);
# 

import tensorflow as tf
import numpy as np

####################### numpy.py ###################


####################################################

a=tf.constant([1.0,2.0],name="a");  #返回张量 Tensor
b=tf.constant([2.0,3.0],name="b");  #返回张量 Tensor
result = a+b;
sess = tf.Session();
sess.run(result);
a_val = sess.run(a);
b_val = sess.run(b);
result_val=sess.run(result);
print("type(a):",type(a));
print("type(b):",type(b));
print("type(result):",type(result));
print("type(a_val):",type(a_val));
print("type(b_val):",type(b_val));
print("type(result_val):",type(result_val));
print("a:",a);
print("a.device:",a.device);
print("a.dtype:",a.dtype);
print("a.graph:",a.graph);
print("a.name:",a.name);
print("a.op:",a.op);
print("a.shape:%s" %(a.get_shape()));
print("a.shape:%s" %(a.shape));
print("b:",b);
print("result:",result);
print("sess:",sess);
#print("a.graph():",a.graph());
#print("b.graph():",b.graph());
print("tf.get_default_graph():",tf.get_default_graph());
