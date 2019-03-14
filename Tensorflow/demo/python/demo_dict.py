#!/usr/bin/env python3 
# -*- coding:utf-8 -*-

############################
#    字典dict使用范例       #
############################
#
# 使用范例:
#   python3 demo_dict.py
############################
#字典定义并赋值
dict1 = {'key1':'val1','key2':2,'key3':['a','b']};
#字典访问
print(dict1.get("key1"));
print(dict1["key1"]);
#字典赋值
dict1["key4"]="val4";
#字典删除
dict1.pop("key1");
#字典大小
print("len(dict1):",len(dict1));
#字典遍历
for key in dict1.keys():
    print("%s:%s" %(key,dict1[key]));
for key,val in dict1.items():
    print("%s:%s" %(key,val));

