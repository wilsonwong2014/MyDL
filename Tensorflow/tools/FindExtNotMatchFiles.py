#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''查找扩展名与类型不匹配文件
    显示不匹配列表：
        $pytnon3 FindExtNotMatchFiles.py --path ~/data/temp --exts '.jpg,.png' --fix 0

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
    description='查找扩展名与类型不匹配文件'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--path',type=str, help='搜索目录. eg. --path "~/data/temp"',default='');
    parser.add_argument('--fix', type=int, help='是否修正(0-不修正,1-修正). eg. --fix 0', default=0);
    parser.add_argument('--exts',type=str, help='搜索扩展名列表. eg. --exts ".jpg,.jpeng,.png"',default='');
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
    fix=arg.fix
    exts=arg.exts
    if not os.path.exists(arg.path):
        print('path:%s not exists!'%(arg.path))
        sys.exit()

    files=funs.FindExtNotMatchFiles(path,exts,fix)
    for sfile in files:
        print(sfile)


if __name__=='__main__':
    arg=params()
    main(arg)

