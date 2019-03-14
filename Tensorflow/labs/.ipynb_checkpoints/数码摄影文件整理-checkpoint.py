#!/usr/bin/env python
# coding: utf-8

# 整理数码摄影文件：
# 1.根据文件名按年月份归类，如文件"IMG_20180105_00001.jpg"归类到 ./demo/2018/01
# 
# 

# In[1]:


#-*- coding: utf-8 -*-
import os
import sys
import shutil
import re
from mylibs import funs
from mylibs.ProcessBar import ShowProcess

#  源目录
src_path='%s/d/数码摄影_src'%(os.getenv('HOME'))
#src_path='%s/data/temp/数码摄影_src'%(os.getenv('HOME'))
#归类目录
dst_path='%s/d/数码摄影'%(os.getenv('HOME'))
#dst_path='%s/data/temp/数码摄影'%(os.getenv('HOME'))


# 创建搜索字典:
#   {匹配关键字:迁移路径}
#   如:
#       {'201805':'/home/hjw/temp/pics/2018/05'}
#   正则匹配表达式:
#       s=r'_201805[0-9]+_'
#       或
#       s=r'_%s[0-9]+_'%(key)

# In[2]:


dicts={'{:0>4d}{:0>2d}'.format(y,m):'{}/{:0>4d}/{:0>2d}'.format(dst_path,y,m) for y in range(2015,2019) for m in range(1,13)}
#for k,v in dicts.items():
#    print('{}:{}'.format(k,v))


# 搜集文件列表

# In[3]:


#files_stat=funs.PathStat(src_path) #=>[目录个数，文件个数，大小{Byte}]
#print(files_stat)
files=[]
funs.GatherFilesEx(src_path,files)
files_num=len(files)
print('files_num:%s'%(files_num))


# 文件分类迁移

# In[4]:


#自动创建分类目录
for k,v in dicts.items():
    if not os.path.exists(v):
        os.makedirs(v)

#文件迁移        
max_steps = 100
pb = ShowProcess(max_steps,'head','tail', 'OK') 
for i,sfile in enumerate(files):
    file_name=os.path.basename(sfile)
    for k,v in dicts.items():
        #正则表达式判断
        if re.search(r'_%s[0-9]+_'%(k),file_name) and os.path.exists(sfile):
            shutil.move( sfile, '{}/{}'.format(v,file_name))  #移动文件或重命名
            if i%1000==0:
                pb.show_process(int(i*100/files_num))
            break
pb.show_process(100) 


# In[ ]:





# In[ ]:




