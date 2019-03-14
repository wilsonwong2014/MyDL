#!/usr/bin/env python3

#ImageNetData数据集
'''ImageNet数据集
官网下载地址：
    http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads

功能描述:
    1.通过ILSVRC2012_ID查询分类信息
    2.通过WNID查询分类信息
    3.返回所有分类信息的DataFrame

分类信息描述：
    info={
         'ID':<ILSVRC2012_ID>,
         'WNID':<WNID>,
         'en_name':<engilish name>,
         'cn_name':<chinese name>,
         'num':<num of train samples>
          }
分类信息元文件meta.mat :    
    分类信息的英文描述，详细查看数据集的 ReadMe,以下为简要说明：
    ------------------
    All information on the synsets is in the 'synsets' array in data/meta.mat.
    To access it in Matlab, type

      load data/meta.mat;
      synsets

    and you will see

       synsets =

       1x1 struct array with fields:
           ILSVRC2012_ID
           WNID
           words
           gloss
           num_children
           children
           wordnet_height
           num_train_images
    ILSVRC2012_ID与WNID一一对应

分类中文信息：
    由在meta.mat同目录下的label_cn.txt文件描述，每行格式如下：
    <WNID>,<cn_name>

'''
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
class ImageNetData(object):
    def __init__(self, data_path):
        self.df,self.labels_cn=self.__LoadData(data_path)
    
    #加载数据
    def __LoadData(self,data_path):
        meta_file='%s/meta.mat'%(data_path)
        label_cn_file='%s/label_cn.txt'%(data_path)
        df=self.__LoadData_meta(meta_file)
        labels_cn=self.__LoadData_label_cn(label_cn_file)
        return df,labels_cn
    
    #加载meta.mat数据
    def __LoadData_meta(self,meta_file):
        meta=loadmat(meta_file) #加载数据文件
        #导入pd.DataFrame数据结构
        index=np.arange(len(meta))                           #索引
        data=meta['synsets']                                 #meta数据集
        df=pd.DataFrame(columns=['ID','WNID','words','num']) #创建空的DataFrame
        for i,d in enumerate(data):
            #构造一条记录
            df_add=pd.DataFrame({
                                'ID':d['ILSVRC2012_ID'][0][0],
                                'WNID':d['WNID'][0],
                                'words':d['words'][0],
                                'num':d['num_train_images'][0][0]
                                })
            #添加记录
            df=df.append(df_add,ignore_index=True)
        return df
    
    
    #加载label_cn数据
    def __LoadData_label_cn(self,label_file):
        #每行数据如：n02110185,哈奇士
        #读取标签文件
        with open(label_file,'r') as f:
            lines=f.readlines()
        #删除两边空格
        lines=[line.strip() for line in lines]
        #标签分割:(WNID,cn_name)
        labels={line[:line.find(',')]:line[line.find(',')+1:] for line in lines}
        return labels
    
    
    #通过ID获取类别信息:训练样本数,WNID,EnName,CnName
    def get_info_from_id(self,id):
        df_sub=self.df.query('ID==%d'%(id))
        if len(df_sub)>0:
            index=df_sub.index[0]
            info={
                'ID':id,
                'WNID':df_sub.loc[index,'WNID'],
                'en_name':df_sub.loc[index,'words'],
                'cn_name':self.labels_cn[df_sub.loc[index,'WNID']],
                'num':df_sub.loc[index,'num']
                }
            return info
        else:
            return None
    
    
    #通过WNID获取类别信息:训练样本数,ID,EnName,CnName
    def get_info_from_wnid(self,wnid):
        df_sub=self.df.query('WNID=="%s"'%(wnid))
        if len(df_sub)>0:
            index=df_sub.index[0]
            info={
                'ID':df_sub.loc[index,'ID'],
                'WNID':df_sub.loc[index,'WNID'],
                'en_name':df_sub.loc[index,'words'],
                'cn_name':self.labels_cn[df_sub.loc[index,'WNID']],
                'num':df_sub.loc[index,'num']
                }
            return info
        else:
            return None
    
    #获取DataFrame
    def get_df(self):
        return self.df

        
#======================
if __name__=='__main__':
    #测试
    import os    
    data_path='%s/e/dataset_tiptical/image_net/ILSVRC2012_devkit_t12/data'%os.getenv('HOME')    
    obj=ImageNetData(data_path)
    info1=obj.get_info_from_id(3)
    info2=obj.get_info_from_wnid('n02110185')
    print(info1)
    print(info2)
