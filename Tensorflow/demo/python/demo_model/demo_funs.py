#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'First string line is comment!'

__author__ = 'wilsonwong'

# 函数模块引用范例:
#
#
#

#普通函数:没有输入参数，没有返回值
def fun1():
    print('function fun1();no arguments,no return value.');

#函数定义：有固定参数，有返回值
def fun2(v1,v2):
    if(not isinstance(v1,float)):
        print('v1 %s is not float!' %(v1));
    if(not isinstance(v2,float)):
        print('v2 %s is not float!' %(v2));
    v = v1 + v2;
    print('function fun2(v1,v2);return v=v1:v2');
    print('%f=fun1(%f,%f)' %(v,v1,v2));


#

if __name__=='__main__':
    fun1();
    fun2(1.1,2.2);
