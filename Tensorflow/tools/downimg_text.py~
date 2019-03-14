#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''搜索文本爬取图像
  搜索文本里面的图像url并下载,文件存储协议:
  第一行为图像URL正则表达式
  第二行...到最后,为搜索文本内容
  范例:
      <images.txt>
      "ObjURL":"(.*?)"
      {"ObjURL":"http:\/\/img0.imgtn.bdimg.com\/it\/u=1653840105,2676195829&fm=214&gp=0.jpg",
      "FromURL":"http:\/\/shanghai.huangye88.com\/xinxi\/9820_196339497.html"},
      {"ObjURL":"http:\/\/img.bimg.126.net\/photo\/tz68kxzapaeymrpyjdp8pg==\/2568740637462204796.jpg",
      "FromURL":"http:\/\/blog.sina.com.cn\/s\/blog_66820bd80100kkn0.html"}],"adType":"0",
      "middleURL":"https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=1653840105,2676195829&fm=26&gp=0.jpg",

  操作步骤:
    1.浏览网页,查看源码,拷贝源码到文本文件 "text.txt".
    2.观察本文,确定图片url匹配正则表达式 "patter".  
    3.把正则表达式,插入到text.txt首行
    4.执行脚本
        $python3 getimg_text.py --infile text.txt --savepath ./temp --prefix img_
 
  说明:
      由于百度url经过加密处理,不适用本脚本处理.

  使用范例:
      $python3 downimg_text.py --savepath ./temp --prefix img_ --infile test.txt
'''

import re
import requests
import os
import itertools
import argparse
import pdb

def downloads(path,prefix,reg,text):
    ''' 搜索文本图片链接并下载
    @param path:本地保存路径
    @param prefix:文件名前缀
    @param reg:文件搜索正则表达式
    @param text:搜索文本
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    i=1
    #regex = re.compile(r'"ObjURL":"(.*?)"')
    #reg='"ObjURL":"(.*?)"'
    regex = re.compile(r'%s'%(reg))
    uu=re.findall(regex, text)
    for downurl in uu:
        url=str(downurl).replace("\\","")
        print('url:',url)
        #获取扩展名
        exts=os.path.splitext(url)
        ext_name=exts[1] if len(exts)==2 else '.jpg'
        try:
            print('download:',url)
            html_1=requests.get(url,timeout=180)
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
    description='搜索文本文件的图像路径并下载'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--savepath',type=str
        , help='保存路径. eg. --savepath "%s/Data/baidu_img"'%(os.getenv('HOME')),default='%s/Data/huitu_img'%(os.getenv('HOME')));
    parser.add_argument('--infile', type=str, help='输入文件路径. eg. --infile "./page.html"', default="");
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
    #读取文件
    if os.path.exists(arg.infile):
        with open(arg.infile,'r') as f:
            content=f.read()
            #分离正则表达式和文本内容
            nPos=content.index('\n')
            if nPos>0:
                reg=content[:nPos]
                text=content[nPos+1:]
                downloads(arg.savepath,arg.prefix,reg,text)
            else:
                print('file content error!')
    else:
        print('file:%s not exists!'%(arg.infile))
    

if __name__=='__main__':
    arg=params()
    main(arg)
