#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''从gif提取png文件
'''

import os
import sys
from PIL import Image
from mylibs import funs
'''
https://blog.csdn.net/huxiangen/article/details/80825181 
'''
sfile='%s/work/data/1.gif'%(os.getenv('HOME'))
to_path='%s/work/temp/%s/png'%(os.getenv('HOME'),sys.argv[0].split('.')[0])
print('gif:',sfile)
print('to_path:',to_path)

if not os.path.exists(to_path):
    os.makedirs(to_path)

im = Image.open(sfile) 
def iter_frames(im): 
    try: 
        i= 0 
        while 1: 
            im.seek(i) 
            imframe = im.copy() 
            if i == 0: 
                palette = imframe.getpalette() 
            else: 
                imframe.putpalette(palette) 
            yield imframe 
            i += 1 
    except EOFError: 
        pass 

for i, frame in enumerate(iter_frames(im)): 
    frame.save('%s/image_%d.png'%(to_path,i),**frame.info)


funs.gif2png(sfile,to_path)
