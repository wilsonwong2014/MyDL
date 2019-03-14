#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''zip函数使用范例
   http://www.runoob.com/python/python-func-zip.html
'''

a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)     # 打包为元组的列表
print(type(zipped))
print(list(zipped))
#    [(1, 4), (2, 5), (3, 6)]
print(list(zip(a,c)))              # 元素个数与最短的列表一致
#    [(1, 4), (2, 5), (3, 6)]
print(list(zip(*zipped)))          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
#    [(1, 2, 3), (4, 5, 6)]

