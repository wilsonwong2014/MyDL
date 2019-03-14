#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################
#   向量表 list使用范例  #
#########################
#
# 使用范例:
#   python3 demo_list.py
#
################################

#定义向量表并赋值
list1 = ['elm1','eml2','elm3'];
#访问
print(list1);
print(list1[0]);
print(list1[-1]);

#追加
list1.append('elm4');
print(list1);

#弹出(删除)末端
elm=list1.pop();
print(elm);

#弹出(删除)指定序号
elm=list1.pop(1);
print(elm);

#插入
list1.insert(1,'new elm1');
list1[1]='new elm2';
list2=[list1,'elm'];
print(list2);

#遍历
for x in list1:
    print(x);

#追加与扩展
list3=[1,2,3]
list4=[4,5,6]
list5=list3.append(list4) #=>[1,2,3,[4,5,6]]    #追加
list6=list3.extend(list4) #=>[1,2,3,4,5,6]      #扩展
