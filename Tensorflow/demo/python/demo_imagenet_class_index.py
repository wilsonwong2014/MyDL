#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''ImageNet中英文对照处理脚本
'''

import json
import os
import sys
import pandas as pd

#imagenet源json文件:{'ID':['folder','EnglishName']}
json_file='/home/hjw/e/dataset_tiptical/imagenet_class_index.json' 
#中英文对照:ID:CnName
cn_name_file='/home/hjw/e/dataset_tiptical/index_cn.txt'
#添加中文翻译的json文件
json_cn_file='/home/hjw/e/dataset_tiptical/imagenet_class_index_cn.json'

'''
temp_file='/home/hjw/e/dataset_tiptical/tmptxt'
with open(temp_file,'w') as f:
    dicts=json.load(open(json_file,'r'))
    for i in range(1000):
        #f.write("'%s':%s\r\n"%(i,dicts[str(i)]))
        f.write('%d:%s\r\n'%(i,dicts[str(i)]))
'''

dicts=json.load(open(json_file,'r'))
df=pd.read_table(cn_name_file,sep=':',header=None)
for i in range(1000):
    dicts[str(i)].append(df.iloc[i,1])

json.dump(dicts,open(json_cn_file,'w'))

for i in range(1000):
    print('%s\r\n'%(dicts[str(i)]))


dicts=json.load(open(json_cn_file,'r'))
