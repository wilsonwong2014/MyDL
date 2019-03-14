#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''字符识别
   https://blog.csdn.net/wwj_748/article/details/78109680?utm_source=tuicool&utm_medium=referral

安装:
  $pip3 install pyocr
  $pip3 install pytesseract
  $sudo apt-get install tesseract-ocr
  $sudo apt-get install tesseract-ocr-chi-sim

  简体中文训练集:
      如果要识别中文需要下载对应的训练集：https://github.com/tesseract-ocr/tessdata 
     ，下载”chi_sim.traineddata”，然后copy到训练数据集的存放路径，如：
 

功能说明:
    图片字符识别,并把结果保存到文件,不支持递归遍历
       文件名:识别结果

参数说明:
    --path      图片存放目录
    --savefile  识别结果保存文件
    --lang      支持语言
使用范例:
    $python3 py_ocr.py --path ./temp --savefile ./temp/result.txt --lang chi_sim
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytesseract
from PIL import Image
import os
import sys
import shutil
import argparse
import pdb

def img2str(img,lang='chi_sim'):
    '''图像转换文字
    @param img:图像数据
    @param lang:语言
    @return s:返回文字转换结果
    '''   
    code = pytesseract.image_to_string(img, lang)
    return code

def imgfile2str(imgfile,lang='chi_sim'):
    '''图像文件转换文字
    @param imgfile:图像文件
    @param lang:语言
    @return s:返回文字转换结果
    '''
    code=None
    try:
        image = Image.open(imgfile)
        code = pytesseract.image_to_string(image, lang='chi_sim')
    except Exception as e:
        print('ocr(%s) err!'%(imgfile))
    return code

def imgs2str(path,lang='chi_sim'):
    '''把目录下的文件转换为对应字符
    @param path:转换目录
    @param lang:语言
    @return {file,code}:转换结果
    '''
    ret={}
    if os.path.exists(path):
        files=os.listdir(path)
        for x in files:
            code=imgfile2str('%s/%s'%(path,x),lang)
            ret[x]=code
    return ret

def params():
    ''' 程序参数
    '''
    #程序描述
    description='批量文件字符转换'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--path',type=str, help='文件目录. eg. --path "./temp"',default='./temp');
    parser.add_argument('--lang', type=str, help='语言. eg. --lang "chi_sim"', default='chi_sim');
    parser.add_argument('--savefile', type=str, help='结果保存文件. eg. --savefile "./temp/result.txt"', default='./temp/result.txt');
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''    
    ret=imgs2str(arg.path,arg.lang)
    if not ret is None:
        with open(arg.savefile,'w') as f:
            for key,val in ret.items():
                f.write('%s:%s'%(key,val))


if __name__=='__main__':
    arg=params()
    main(arg)
    print('Finished!')

