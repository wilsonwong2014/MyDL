#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 文件重命名
   把指定目录下的文件名,重新按序号命名,支持递归,每个子目录单独处理

功能说明:
    前缀加流水号重命名文件,支持递归.

参数说明:
    --from_path 
使用范例:
    $python3 rename_index.py --path ./path --prefix new_ --mask_num num.txt
    ./path/1/asd.jpg
    ./path/1/ff.jpg
    ./path/2/fas.jpg
    ./path/2/ffa.jpg
    =>
    ./path/1/img_0.jpg
    ./path/1/img_1.jpg
    ./path/1/num.txt =>2
    ./path/2/img_0.jpg
    ./path/2/img_1.jpg
    ./path/2/num.txt =>2
'''
import os
import sys
import shutil
import argparse
import pdb

#文件重命名，单个目录处理
def rename_index(path,prefix='',mask_num='num.txt'):
    ''' 文件重命名
    @param path:目录
    @param prefix:新命名前缀
    @param mask_num:是否生成 num.txt 标记 文件数
    '''
    if not os.path.exists(path):
        print('path:"%s" not exists!'%(path))
    else:
        files=os.listdir(path)
        index=0
        for src in files:
            src_file=os.path.join(path,src)
            if os.path.isfile(src_file) and src!=mask_num:
                exts=os.path.splitext(src)
                dst_file='%s/%s%d%s'%(path,prefix,index,exts[1])
                shutil.move( src_file, dst_file)
                index+=1
        #标记文件数
        if mask_num!='':
            with open('%s/%s'%(path,mask_num),'w') as f:
                f.write(str(index))

#文件重命名，递归目录
def rename_indexs(path,prefix='',mask_num=False):
    files=os.listdir(path)
    for sfile in files:
        sub_path=os.path.join(path,sfile)
        if os.path.isdir(sub_path):
            rename_indexs(sub_path,prefix,mask_num)
            rename_index(sub_path,prefix,mask_num)


def params():
    ''' 程序参数
    '''
    #程序描述
    description='文件重命名'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--path',type=str, help='目录. eg. --path "./temp"',default='');
    parser.add_argument('--prefix', type=str, help='文件名前缀. eg. --prefix "new_"', default='');
    parser.add_argument('--mask_num', type=str, help='标记文件数. eg. --mask_num "num.txt"', default='');
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    rename_indexs(arg.path,arg.prefix,arg.mask_num)

if __name__=='__main__':
    arg=params()
    main(arg)
    print('Finished!')

