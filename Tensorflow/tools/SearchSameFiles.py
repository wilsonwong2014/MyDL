#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''相同文件处理
    相同文件：内容完全一样，但文件名可能不一样的文件，通过文件大小和文件前后若干字节的md5进行识别
    显示相同文件：
        $pytnon3 SearchSameFiles.py --path ~/data/temp --exts '.jpg,.png'

    删除相同文件：
        $python3 SearchSameFiles.py --path ~/data/temp --fun 1 '.jpg,.png'

    迁移相同文件：
        $python3 SearchSameFiles.py --path ~/data/temp --fun 2 --move_to '~/data/temp_to' --exts '.jpg,.png'

'''

import os
import sys
import argparse
import pdb

from mylibs.SearchSameFiles import SearchSameFiles

def params():
    ''' 程序参数
    '''
    #程序描述
    description='相同文件处理'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--path',type=str, help='搜索目录. eg. --path "~/data/temp"',default='');
    parser.add_argument('--fun', type=int, help='功能(0-显示搜索结果,1-删除相同文件,2-迁移相同文件). eg. --fun 0', default=0);
    parser.add_argument('--move_to',type=str, help='相同文件迁移目录,fun==2时有效. eg. --move_to "~/data/temp_to"',default='');
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
    move_to=arg.move_to
    exts=arg.exts
    if not os.path.exists(arg.path):
        print('path:%s not exists!'%(arg.path))
        sys.exit()

    
    if arg.fun==0:
        print('相同文件查询')
        #显示列表
        obj=SearchSameFiles(path,exts)
        files_same=obj.GetSameFiles()
        for files in files_same:
            print('---------------------')
            for sfile in files:
                print(sfile)
        print('finished!')
    elif arg.fun==1:
        print('删除相同文件')
        #删除相同文件
        obj=SearchSameFiles(path,exts)
        obj.DelSameFiles()
        print('finished!')
    elif arg.fun==2:
        print('迁移相同文件')
        #迁移相同文件
        obj=SearchSameFiles(path,exts)
        obj.MoveSameFiles(move_to)
        print('finished!')
    else:
        print('未知指令：fun(%d)'%(arg.fun))

if __name__=='__main__':
    arg=params()
    main(arg)

