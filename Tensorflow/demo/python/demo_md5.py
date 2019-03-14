#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

'''md5
'''

import hashlib
m2=hashlib.md5()
m2.update('str'.encode())
print("str'md5 is :",m2.hexdigest())
