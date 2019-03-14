#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#####################
#  if使用范例       #
#####################
#
# 使用范例:
#   python3 demo_if.py
#########################

age = 3;
if age>8:
    print('%d>8' %(age));
elif age>5:
    print('%d>8' %(age));
else:
    print('%d<8' %(age));

ret=True if age>3 else False
print('ret:',ret)
