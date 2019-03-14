#!/usr/bin/env python3 
# -*- coding:utf-8 -*-

#############################
#     集合 set 使用范例      #
#############################
#
# 使用范例:
#    python3 demo_set.py
#
##############################

#集合定义并赋值
set1=set([1,2,3]);
set2=set([3,4,5]);
#打印集合内容
print("set1:",set1);
print("set2:",set2);
#集合长度
print("len(set1):",len(set1));
print("len(set2):",len(set2));
#添加
set1.add(4);
set2.add(6);
#删除
set1.remove(1);
#交集
set3=set1 & set2;
#并集
set4=set1 | set2;
#遍历
for x in set1:
    print(x);

