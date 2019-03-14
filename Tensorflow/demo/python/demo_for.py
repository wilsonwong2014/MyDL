#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#############################
#    演示for循环使用方法     #
#############################
#
# 使用范例:
#   python3 demo_for.py
#############################

#=====切片======
arr=range(10);     #0,1,2,3,4,5,6,7,8,9
arr_sub=arr[:5];   #0,1,2,3,4
arr_sub=arr[1:5];  #1,2,3,4
arr_sub=arr[-1] ;  #9
arr_sub=arr[-2:];  #8,9
arr_sub=arr[-2:-1];#8
arr_sub=arr[:8:2]; #0,2,4,6
arr_sub=arr[:];    #0,1,2,3,4,5,6,7,8,9
arr_sub=arr[::2];  #0,2,4,6,8


#迭代１
list1 = ['a','b','c'];
for x in list1:
    print(x);

#迭代２
tuple1 = ('a','b','c');
for x in tuple1:
    print(x);

#迭代３
for x in [1,2,3,4]:
    print(x);

#迭代４
for x in range(10):
    print(x);

dict1={'key1':'val1','key2':'val2','key3':'val3','key4':'val4'};
#迭代５
for key in dict1:
    print(key);# print KeyName=>key1,key2,key3,key4

#迭代６
for val in dict1.values():
    print(val); # print Value=>val1,val2,val3,val4

#迭代７
for k,v in dict1.items():
    print(k,'=',v); #print Key=val => key1=val1,key2=val2,key3=>val3,key4=val4

#迭代８
for i, value in enumerate(['A', 'B', 'C']):
    print(i, value); #0 A,1 B,2 C

#迭代９
for x, y in [(1, 1), (2, 4), (3, 9)]:
    print(x, y) #1 1,2 4,3 9

#生成列表1
list1=[x * x for x in range(1, 11)]; #=>[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

#生成列表２
list1=[x * x for x in range(1, 11) if x % 2 == 0];#=>[4, 16, 36, 64, 100]

#生成列表３
list1=[m + n for m in 'ABC' for n in 'XYZ'];#=>['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']

#生成器１
g = (x * x for x in range(10)); #取值:next(g)
for x in g:
    print(x);

#生成器２　- yield
#  关键字yield特性：遇到yield返回，再次执行，接着上次语句执行
#使用范例:
def my_field_fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b;
        a, b = b, a + b;
        n = n + 1;
    return 'done';

def my_yield_test():
    for x in my_field_fib(10):
        print("my_yield_test:",x);


n = 10;
while n>=0:
    print(n);
    n = n-1;


