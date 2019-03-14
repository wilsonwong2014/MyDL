#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

def Add(d):
    d=d+1
    print(d)
    return d

data=list(range(0,64*2))
hogdata = [map(Add,row) for row in data]#
#hogdata=list(hogdata)
#trainData = np.float32(hogdata).reshape(-1,64)

print(list(map(Add,data)))
data2=[map(Add,data)]
print(data)
print(list(data))
print(type(data2))
print(type(list(data2)))

print(list(data2[:]))
