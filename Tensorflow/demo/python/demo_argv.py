#!/usr/bin/env python3
# -* coding: utf-8 -*-

########################
# 命令行参数解析 argv   #
########################
# 引用 sys,模式与C++的 int main(int argc,char **argv)类似
#
# 使用范例:
#  python3 demo_argv.py a b c d
#
##########################################################

import os;
import sys;

#参数个数
argc = len(sys.argv);
print("argc:",argc);
#打印所有参数
for x in range(0,argc):
    print("argv[%s]:%s" %(x,sys.argv[x]));

