#!/usr/bin/env python3
# -*- coding: utf-8 -*-

list1 = ['elm1','eml2','elm3'];
print(list1);
print(list1[0]);
print(list1[-1]);
list1.append('elm4');
print(list1);
elm=list1.pop();
print(elm);
elm=list1.pop(1);
print(elm);
list1.insert(1,'new elm1');
list1[1]='new elm2';
list2=[list1,'elm'];
print(list2);

tuple1=('elm1','elm2');
print(tuple1);

