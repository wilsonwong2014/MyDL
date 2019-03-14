#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
''' json序列化与序列化范例
        http://www.runoob.com/python/python-json.html
    使用 JSON 函数需要导入 json 库：import json。
        json.dumps 	将 Python 对象编码成 JSON 字符串
        json.loads	将已编码的 JSON 字符串解码为 Python 对象
'''

import os
import sys
import json

#json文件
json_file='%s/work/data/gtest/1.json'%(os.getenv('HOME'))
print('json_file:',json_file)

#构造对象
lst=[]
for n in range(10):
    dic={'src':n,'dst':n*10}
    lst.append(dic)

#json编码
json.dump(lst,open(json_file,'w'))
#json解码
lst2=json.load(open(json_file,'r'))

print('lst:',lst)
print('lst2:',lst2)
