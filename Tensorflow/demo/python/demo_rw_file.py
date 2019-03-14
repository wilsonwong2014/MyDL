#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python 文件读写操作操作
    https://www.cnblogs.com/ghsme/p/3281169.html
'''

import os;
import sys;
############# 调试 begin ###############
argc = len(sys.argv);
import pdb       
if argc>1 and sys.argv[1]=='dbg':    
    pdb.set_trace(); #调试
############# 调试 end   ###############

import struct
import numpy as np

########################################
#文本文件操作
def demo_text():
    filepath='demo_rw_file.txt';
    #写方式打开文件
    print('Write text file!');
    with open(filepath,'w') as f:
        #写入内容 
        f.write('write one line 1.\r\n');
        f.write('\r\n');
        f.write('write one line 2.');
        #关闭文件
    #--------------------------
    #一次性读取所有文件内容
    print('Read all content!');
    with open(filepath,'r') as f:
        str=f.read();
        print(str);
    
    #--------------------------
    #每次读取指定大小内容
    print('Read special size content!');
    with open(filepath,'r') as f:
        str=None;
        index=0;

        while(True):
            str=f.read(5);
            if(len(str)<=0):
                break;
            print('Index:[%d]%s' %(index,str));
            index=index+1;

    #--------------------------
    #每次读取一行内容
    print('Read one line!');
    with open(filepath,'r') as f:
        str=None;
        index=0;
        while(True):
            str=f.readline();
            if(len(str)<=0):
                break;
            print('Index:[%d]%s' %(index,str));
            index=index+1;


#==============================
#二进制文件操作
def demo_bin():
    filepath='demo_rw_file.bin';
    #写方式打开文件
    print('write for binary!');
    with open(filepath,'wb') as f:
        val_int8=bytes([1]);
        val_int16=2;
        val_int32=3;
        val_int64=4;
        val_float32=5.0;
        val_float64=6.0;
        f.write(struct.pack('>chiqfd',val_int8,val_int16,val_int32,val_int64,val_float32,val_float64));

    #读方式打开文件
    print('read for binary!');
    with open(filepath,'rb') as f:
        val_int8=bytes(0);
        val_int16=0;
        val_int32=0;
        val_int64=0;
        val_float32=0;
        val_float64=0;
        (val_int8,val_int16,val_int32,val_int64,val_float32,val_float64)=struct.unpack('>chiqfd',f.read(27));
        print('val_int8:%s,val_int16:%d,val_int32:%d,val_int64:%d,val_float32:%f,val_float64:%f' \
            %(val_int8,val_int16,val_int32,val_int64,val_float32,val_float64));

    #np.fromfile读取
    print('read for np.fromfile');
    with open(filepath,'rb') as f:
        val_bytes = np.fromfile(f,dtype=np.uint8);
        print(val_bytes);

############## 测试 ###############
if __name__ == '__main__':
    demo_text();
    demo_bin();

    
