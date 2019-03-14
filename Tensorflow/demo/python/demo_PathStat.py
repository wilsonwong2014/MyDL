#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''PathStat范例
'''
import os
import sys
#import pdb
#pdb.set_trace()
from mylibs import funs

if len(sys.argv)!=2:
    print('usge:%s path'%(sys.argv[0]))
else:
    info=funs.PathStat(sys.argv[1])
    print('dirs,files,size:',info)
