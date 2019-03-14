#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 命令行参数解析
    https://www.cnblogs.com/zz22--/p/7719285.html
    #命名行范例
    python get.py -h -o 1 --help --mode=sift file1 file2
    #参数解析
    opts,args=getopt.getopt(sys.argv[1:],'ho:',['help','mode=']
    
    短格式:参数只有一个字符,前面加-,如 -h ,-o
    长格式:参数为一个单词,前面加--,如 --help, --mode
    有附加参数(短格式): 参数后面加":",如"o:",参数与附加参数之间可以有空格或没空格,如 -o1 -o 1 -oabc
    有附加参数(长格式):参数后面加"=",如"mode=",=两边不能有空格,如 --mode=sift
    格式化参数通过 opts返回:[("-h",""),("-o","1"),("--help",""),("--mode","sift")]
    剩余非格式参数通过 args返回:["file1","file2"]
'''

import sys
import getopt

opts,args=getopt.getopt(sys.argv[1:],'ho:',['help','mode='])
print('opts:',opts)
print('args:',args)
#参数处理
for key,val in opts:
    if key in('-h','--help'):
        print('Switch:',key)
    if key in('-o','--mode'):
        print('key:%s,val:%s' %(key,val))


