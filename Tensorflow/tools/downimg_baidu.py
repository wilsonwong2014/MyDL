#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 百度图像爬取
    http://blog.51cto.com/11623741/2097160

    使用范例:
    $python3 downimg_baidu.py --savepath ./temp --keys "手写数字" --prefix img_ --max_pages 10 --width 1024 --height 768
    ========
    python3 downimg_baidu.py --savepath ./temp/cat --keys "猫" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/cat --keys "猫" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/dog --keys "狗" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/dog --keys "狗" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/cow --keys "牛" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/cow --keys "牛" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/sheep --keys "羊" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/sheep --keys "羊" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/pig --keys "猪" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/pig --keys "猪" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/chick --keys "鸡" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/chick --keys "鸡" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/rabbit --keys "兔" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/rabbit --keys "兔" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/bear --keys "熊" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/bear --keys "熊" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/monkey --keys "猴" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/monkey --keys "猴" --max_pages 200 --width 224 --height 224
    python3 downimg_baidu.py --savepath ./temp/mouse --keys "鼠" --max_pages 200 --width 200 --height 200
    python3 downimg_baidu.py --savepath ./temp/mouse --keys "鼠" --max_pages 200 --width 224 --height 224
'''

import json
import itertools
import urllib
import requests
import os
import re
import sys
import argparse
import pdb

def downloads(path,word,prefix='',max_pages=10,width='',height=''):
    '''百度图片爬取
    @param path:本地保存路径
    @param word:搜索关键字
    @param prefix:文件名前缀
    @param max_pages:搜索页数,每页60项
    @param width:图像宽度
    @param height:图像高度
    '''
    #word='手写数字'
    #path='%s/Data/Temp/img1'%(os.getenv('HOME'))
    if not os.path.exists(path):
        os.mkdir(path)
    word=urllib.parse.quote(word)
    #该URL地址不是网页的URL地址，而JSON地址
    url = r"http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&ic=0&width={width}&height={height}&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
    if max_pages<=0:
        max_pages=10
    urls=[url.format(word=word,width=width,height=height,pn=x*60)for x in range(0,max_pages)]
    index=0
    str_table = {
        '_z2C$q': ':',
        '_z&e3B': '.',
        'AzdH3F': '/'
    }

    char_table = {
        'w': 'a',
        'k': 'b',
        'v': 'c',
        '1': 'd',
        'j': 'e',
        'u': 'f',
        '2': 'g',
        'i': 'h',
        't': 'i',
        '3': 'j',
        'h': 'k',
        's': 'l',
        '4': 'm',
        'g': 'n',
        '5': 'o',
        'r': 'p',
        'q': 'q',
        '6': 'r',
        'f': 's',
        'p': 't',
        '7': 'u',
        'e': 'v',
        'o': 'w',
        '8': '1',
        'd': '2',
        'n': '3',
        '9': '4',
        'c': '5',
        'm': '6',
        '0': '7',
        'b': '8',
        'l': '9',
        'a': '0'
    }
    i=1
    char_table = {ord(key): ord(value) for key, value in char_table.items()}
    for url in urls:
        print('page:',url)
        html=requests.get(url,timeout=10).text
        #设置编译格式
        a=re.compile(r'"objURL":"(.*?)"')
        downURL=re.findall(a,html)
        for t in downURL:
            #解码
            for key, value in str_table.items():
                t = t.replace(key, value)
            t=t.translate(char_table)
            #获取扩展名
            exts=os.path.splitext(t)
            ext_name=exts[1] if len(exts)==2 else '.jpg'
            try:
                print('download:',t)
                html_1=requests.get(t,timeout=180)
                if str(html_1.status_code)[0]=="4":
                    print('失败1')
                    continue   
            except Exception as e:
                print('失败2')
                continue
            #下载
            with open(path+"/"+prefix+str(i)+ext_name,'wb') as f:
                f.write(html_1.content)
            i=i+1


def params():
    ''' 程序参数
    '''
    #程序描述
    description='百度图像爬取'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--savepath',type=str
        , help='保存路径. eg. --savepath "%s/Data/baidu_img"'%(os.getenv('HOME')),default='%s/data/baidu_img'%(os.getenv('HOME')));
    parser.add_argument('--keys',type=str, help='搜索关键字. eg. --keys "手写数字"',default='手写数字');
    parser.add_argument('--max_pages', type=int, help='最大下载页数(0-不限制,默认10). eg. --max_pages 10', default=10);
    parser.add_argument('--width', type=str, help='图像宽度. eg. --width ""', default="");
    parser.add_argument('--height', type=str, help='图像高度. eg. --height ""', default="");
    parser.add_argument('--prefix', type=str, help='文件名前缀. eg. --prefix "img1_"', default="");
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    downloads(arg.savepath,arg.keys,arg.prefix,arg.max_pages,arg.width,arg.height)

if __name__=='__main__':
    arg=params()
    main(arg)

