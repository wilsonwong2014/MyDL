#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''搜索相同文件
    from mylibs.SearchSameFiles import SearchSameFiles
    obj=SearchSameFiles('/path','.jpg,.png')    
    检索相同文件
        files_same=obj.GetSameFiles()
    删除相同文件
        obj.DelSameFiles()
    迁移相同文件
        obj.MoveSameFiles('/to_path')
'''
import pdb
#pdb.set_trace()

import os
import sys
import shutil
import hashlib
import pandas as pd
import numpy as np
from mylibs import funs
from mylibs import ProcessBar


class SearchSameFiles(object):
    #构造函数
    def __init__(self,path,exts=''):
        self.path=path     #搜索目录
        self.exts=exts     #扩展名，如:".jpg,.png"
        self.pb=ProcessBar.ShowProcess(100,'','','') #进度条
        self.info_path=funs.PathStat(path)            #目录信息统计

    #获取文件特征：大小，md5
    def get_feature(self,sfile):
        fsize=os.path.getsize(sfile)
        smd5=[]
        with(open(sfile,'rb')) as f:
            #读取前面50个字节
            bytes=f.read(50)
            #读取后面50个字节
            f.seek(50,2)#倒数50个字节
            bytes+=f.read(50)
            #md5
            m5=hashlib.md5()
            m5.update(bytes)
            smd5=m5.hexdigest()
        return (sfile,fsize,smd5)


    #检索相同文件
    def GetSameFiles(self):
        #构造数据表
        df=pd.DataFrame(columns=['sfile','fsize','md5'])
        #收集文件
        print('收集文件列表')
        files=funs.GatherFiles(self.path,exts=self.exts)
        print('计算文件特征')
        pb=ProcessBar.ShowProcess(100,'','', infoDone = 'Done')
        #计算文件特征
        nFiles=len(files)
        for i, sfile in enumerate(files):
            #获取文件特征
            feats=self.get_feature(sfile)
            #添加一行
            df.loc[i]=feats
            if i%50==0:
                pb.show_process(int(i*100/nFiles))
        pb.show_process(100)
        #汇总
        files_same=[]
        df_group=df.groupby(['fsize','md5'])
        for name,group in df_group:
            if group.shape[0]>1:
                files_same.append(group.loc[:,'sfile'])
        return files_same


    #删除相同文件
    def DelSameFiles(self):
        #检索相同文件
        files_same=self.GetSameFiles()
        print('删除相同文件')
        pb=ProcessBar.ShowProcess(100,'','', infoDone = 'Done')
        #删除相同文件
        nGroups=len(files_same)
        for i,files_sub in enumerate(files_same):
            for sfile in files_sub[1:]:
                os.remove(sfile)
            pb.show_process(int(i*100/nGroups))
        pb.show_process(100)
                
    #拷贝相同文件
    def MoveSameFilesTo(self,to_path):
        #检索相同文件
        files_same=self.GetSameFiles()
        print('迁移相同文件:',to_path)
        pb=ProcessBar.ShowProcess(100,'','', infoDone = 'Done')
        #拷贝相同文件
        nGroups=len(files_same)
        src_path_len=len(self.path)
        for i, files_sub in enumerate(files_same):
            for sfile in files_sub[1:]:
                src_file=sfile
                dst_file='%s%s'%(to_path,sfile[src_path_len:])
                dst_path=os.path.split(dst_file)[0]
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                print('src:',src_file)
                print('dst:',dst_file)
                shutil.move(src_file,dst_file)
            pb.show_process(int(i*100)/nGroups)
        pb.show_process(100)

#===================
if __name__=='__main__':
    #usge: 
    if len(sys.argv)!=4:
        print('usge:%s path exts to_path'%(sys.argv[0]))
    else:
        path=sys.argv[1]
        exts=sys.argv[2]
        to_path=sys.argv[3]
        obj=SearchSameImgs(path,exts)
        #GetSameFiles
        files_same=obj.GetSameFiles()
        for files in files_same:
            print('---------------------')
            for sfile in files:
                print(sfile)
        #------------------------
        #obj.MoveSameFilesTo(to_path)
        #obj.DelSameFiles()

    

