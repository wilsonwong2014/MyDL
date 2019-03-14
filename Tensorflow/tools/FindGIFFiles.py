#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''查找GIF文件
    $pytnon3 FindGIFFIles.py --path ~/data/temp --delflag 0 --unzipflag 0

'''

import os
import sys
import argparse
import pdb

from mylibs import funs

def params():
    ''' 程序参数
    '''
    #程序描述
    description='查找gif文件'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--path',type=str, help='搜索目录. eg. --path "~/data/temp"',default='');
    parser.add_argument('--delflag', type=int, help='删除标记 1-删除. eg. --delflag 0', default=0);
    parser.add_argument('--unzipflag',type=int, help='提取png标记，1-提取. eg. --unzipflag 0',default=0);
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    path=arg.path
    if not os.path.exists(arg.path):
        print('path:%s not exists!'%(arg.path))
        sys.exit()

    files=funs.FindGIFFiles(arg.path,arg.delflag,arg.unzipflag)
    for sfile in files:
        print(sfile)    
    
if __name__=='__main__':
    arg=params()
    main(arg)

