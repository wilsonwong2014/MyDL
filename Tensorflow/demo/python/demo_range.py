#!/usr/bin/env python3
# -*- codeing:utf-8 -*-

##########################
#     range函数使用      #
##########################
#定义原型:
#  class range(stop) 
#  class range(start, stop[, step]) 
#
#使用范例:
#>>> list(range(10))
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#>>> list(range(1, 11))
#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#>>> list(range(0, 30, 5))
#[0, 5, 10, 15, 20, 25]
#>>> list(range(0, 10, 3))
#[0, 3, 6, 9]
#>>> list(range(0, -10, -1))
#[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
#>>> list(range(0))
#[]
#>>> list(range(1, 0))
#[]
#
##########################################

print("range(10):");
print(list(range(10)));        #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print("range(1,11):");
print(list(range(1, 11)));     #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("range(0,30,5):");
print(list(range(0, 30, 5)));  #[0, 5, 10, 15, 20, 25]

print("range(0,10,3):");
print(list(range(0, 10, 3)));  #[0, 3, 6, 9]

print("range(0,-10,-1):");
print(list(range(0, -10, -1)));#[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]

print("range(0):");
print(list(range(0)));         #[]

print("range(1,0):");
print(list(range(1, 0)));      #[]

