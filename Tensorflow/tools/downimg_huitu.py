#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 绘图网爬虫
    http://blog.51cto.com/11623741/2097160

    使用范例:
    $python3 downimg_huitu.py --savepath ./temp --keys "手写数字" --prefix img_ --max_pages 10
'''
import re
import requests
import os
import sys
import urllib
import argparse
import pdb

def downloads(path,word,prefix='',max_pages=10,width='',height=''):
    '''绘图网爬虫
    @param path:本地保存路径
    @param word:搜索关键字
    @param prefix:文件名前缀
    @param max_pages:搜索页数,每页60项
    @param width:图像宽度
    @param height:图像高度
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    header= {'content-type': 'application/json',
           'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}
    url="http://soso.huitu.com/Search/GetAllPicInfo?perPageSize=102&kw={word}&page={num}"
    if max_pages<=0:
        max_pages=10
    word=urllib.parse.quote(word)
    urls=[str(url).format(word=word,num=x)for  x in  range(1,max_pages)]
    i=1
    for url in urls:
        print('page:',url)
        html=requests.get(url).text
        #print(html)
        r=re.compile(r'"imgUrl":"(.*?)"')
        u=re.findall(r,html)
        for s in u:
            #获取扩展名
            exts=os.path.splitext(s)
            ext_name=exts[1] if len(exts)==2 else '.jpg'
            try:
                print('download:',s)
                html_1=requests.get(s,timeout=180)
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
    description='绘图爬虫'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--savepath',type=str
        , help='保存路径. eg. --savepath "%s/Data/baidu_img"'%(os.getenv('HOME')),default='%s/Data/huitu_img'%(os.getenv('HOME')));
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
