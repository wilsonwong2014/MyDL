#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' 百度图像爬取
    http://blog.51cto.com/11623741/2097160
    通过json配置：
        json配置文件语法：
        "savepath":"/temp"          -------保存路径
        "keys":"猫,狗"              -------搜索关键字
        "exts":".jpg,.jpeg,.png"    -------下载扩展名列表
        "z":0                       -------图像大中小标志[与百度语法同步]
        "width":0                   -------图像宽度
        "height":0                  -------图像高度
        "prefix":""                 -------文件名前缀
        "down_num":0                -------下载个数,默认600
        "dbfile":"/temp/db.json"    -------已下载文件列表[格式另说明],空：默认文件名db.json存放在key保存目录
    下载文件列表json说明：
        {
            down_num:{key:num},     ------关键字下载状态，0-未下载完成，1-下载完成
            files   :{url:local}    ------下载列表
        }
        范例：
        {
            "down_num":{"猫":10,"狗":20}
            "files":{"url1":"/dir/f1.jpg","url2":"/dir/f2.jpg"}
        }
    json配置范例：
        {
        "savepath":"/home/hjw/data/baidu_img"
        "keys":"猫,狗"
        "exts":"jpg,jpeg,png"
        "z":0
        "width":0
        "height":0
        "prefix":""
        "max_pages":0
        "dbfile":"/home/hjw/baidu_img/db.json"
        }
    规则：
        1.下载图像保存根目录为：config.savepath
        2.每个关键字保存目录为根目录下以关键字命名的子目录
        3.config.dbfile记录已下载图像url以供查询，避免重复下载；
            如果为空：各关键字有独立查询dbfile,默认为该关键字命名的目录下db.json
            如果指定路径：每个json配置文件下的各关键字共享一个dbfile
    使用范例:
        #下载(单个json配置)
            $python3 downimg_baidu.2.py --opt ~/e/json_down/config_baidu.0.json
        #下载(批json下载)
            $python3 downimg_baidu.2.py --opt ~/e/json_down
        #批量构造json配置文件
            $python3 downimg_baidu.2.py --fun 1 --save_path '~/data/temp/imgs' --json_path '~/data/temp/json' --keys_file '~/data/temp/json/keys.txt'
            $python3 downimg_baidu.2.py --fun 1 --save_path ~/e/dataset_crawl --json_path ~/e/json_down --keys_file ~/e/json_down/keys.txt

'''

import json
import itertools
import urllib
import requests
import imghdr
import os
import re
import sys
import multiprocessing
import argparse
import pdb


#百度下载
class BaiduDownload(object):
    def __init__(self):
        #-----------编码表-----------
        self.str_table = {
            '_z2C$q': ':',
            '_z&e3B': '.',
            'AzdH3F': '/'
        }
        self.char_table = {
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
        self.char_table = {ord(k): ord(v) for k, v in self.char_table.items()}        


    #判断是否已下载
    def IsDowned(self,dictDowns,url):
        files=dictDowns.get('files')
        if files!=None:
            local_file=files.get(url)
            if local_file != None:
                return True
        return False


    #url解码
    def url_translate(self,url):    
        #解码
        for k, v in self.str_table.items():
            url = url.replace(k, v)
        url=url.translate(self.char_table)
        return url


    #下载
    def download_url(self,url,local_file,key_path):
        '''下载图像
        @param url          ---下载图像url
        @param local_file   ---下载到本地路径
        @param str_table    ---编码表
        @param char_table   ---编码表
        '''
        #下载
        try:
            html_1=requests.get(url,timeout=10)
            if html_1.content[0:3]==b'GIF':
                print('gif not support!')
                return -1
            if str(html_1.status_code)[0]=="4":
                print('失败[404]')
                return -1
        except Exception as e:
            print('失败:\r',e)
            return -1
        #本地路径保存
        with open(local_file,'wb') as f:
            f.write(html_1.content)
            return 0
        #end {with open(local_file,'wb') as f:}
        return -1

    #获取本地下载路径
    def GetLocalFilePath(self,url,key_path,nStartNo):
        while 1:
            local_file='%s/%d%s'%(key_path,nStartNo,os.path.splitext(url)[1])
            if not os.path.exists(local_file):
                return local_file
            nStartNo+=1

    #下载
    def downloads(self,cfg_file):
        '''百度图片爬取
        '''
        #---------加载json配置文件-------
        config=json.load(open(cfg_file,'r'))
        #json配置
        z=config.get('z')
        if z==None:
            z=1
        width=config.get('width')
        if width==None:
            width=0
        height=config.get('height')
        if height==None:
            height=0
        save_path=config.get('savepath')
        dbfile=config.get('dbfile')
        keys=config.get('keys')
        prefix=config.get('prefix')
        exts=config.get('exts')
        down_num=config.get('down_num')
        if down_num==None:
            down_num=600
        max_pages=down_num//60+5
        time_out=config.get('time_out')
        if time_out == None:
            time_out=20
        
        #---------加载已下载列表--------
        dictDowns={}
        if os.path.exists(dbfile):
            dictDowns=json.load(open(dbfile,'r'))
        if dictDowns.get('files')==None:
            dictDowns['files']={}

        #搜索队列中的关键字
        print('=========================')
        print('json config:%s'%(cfg_file))
        print(config)
        print('=========================')
        for key in keys.split(','):
            print('---------key:%s---------'%(key))
            key_break=False
            #关键字保存目录检测
            key_path=os.path.join(save_path,key)
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            #统计当前目录文件数
            nFiles=len(os.listdir(key_path))
            #下载数据库路径
            if dbfile==None or dbfile=='':
                dbfile='%s/db.json'%(key_path)
            word=urllib.parse.quote(key)
            #该URL地址不是网页的URL地址，而JSON地址
            url = r"http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&st=-1&z={z}&ic=0&width={width}&height={height}&word={word}&face=0&istype=2nc=1&pn={pn}&rn=60"
            urls=[url.format(z=z,word=word,width=width,height=height,pn=x*60)for x in range(0,max_pages)]
            print('--------------------------')
            print('[%s]page count:%d'%(key,len(urls)))
            print('--------------------------')
            index=0
            i=1
            for page_url in urls:
                print('[%s]page:%s'%(key,page_url))
                html=requests.get(page_url,timeout=time_out).text
                #设置编译格式
                a=re.compile(r'"objURL":"(.*?)"')
                downURL=re.findall(a,html)
                print('--------------------------')
                print('[%s]url count:%d'%(key,len(downURL)))
                print('--------------------------')
                for url in downURL:
                    url=self.url_translate(url)
                    print('[%s]url:%s'%(key,url))
                    #扩展名
                    ext_name=os.path.splitext(url)[1].lower()
                    #排除扩展名
                    if (ext_name=='' or not ext_name in exts) and exts!="":
                        print('{ext_name:%s} not support{%s}! '%(ext_name,exts))
                        continue
                    #检查是否已下载
                    if self.IsDowned(dictDowns,url):
                        print('exists!')
                        continue
                    local_file=self.GetLocalFilePath(url,key_path,nFiles)
                    ret=self.download_url(url,local_file,key_path)
                    if ret==1:
                        key_break=True
                        break;
                    print('[%s]local_file:%s'%(key,local_file))
                    dictDowns['files'][url]=local_file
                    nFiles+=1
                    if nFiles%5==0:
                        json.dump(dictDowns,open(dbfile,'w'))
                if key_break:
                    json.dump(dictDowns,open(dbfile,'w'))
                    break;
                #end {for url in downURL:}
            #end {for key in keys.split(','):}

    #批量下载{多进程}
    def downloads_path(self,path):
        #检索所有json文件下载
        files=os.listdir(path)
        pool = multiprocessing.Pool(processes = 10)#同时并发10个进程
        for i,sfile in enumerate(files):
            if os.path.splitext(sfile)[1]=='.json':
                json_file=os.path.join(path,sfile)
                print('create process(%d):%s'%(i,json_file))
                pool.apply_async(self.downloads,(json_file,))
        pool.close()
        pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
        print("Sub-process(es) done.")

    #批量生成json配置文件
    def create_json_config(self,save_path,json_path,keys_file):
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        with open(keys_file,'r') as f:
            keys=f.read()
            keys=keys.split(',')
            for i, key in enumerate(keys):
                cfg={}
                cfg['savepath']=save_path
                cfg['keys']=str(key)
                cfg['exts']='.jpg,.jpeg,.bmp,.png'
                cfg['z']=1
                cfg['width']=0
                cfg['height']=0
                cfg['prefix']=''
                cfg['down_num']=1400
                cfg['time_out']=20
                cfg['dbfile']=''
                json_file='%s/config_baidu.%d.json'%(json_path,i)
                json.dump(cfg,open(json_file,'w'),ensure_ascii=False) #参数ensure_ascii:true,把汉字转换为unicode编码，false:汉字输出
               


def params():
    ''' 程序参数
    '''
    #程序描述
    description='百度图像爬取'
    # Create ArgumentParser() object
    parser = argparse.ArgumentParser(description=description);
    # Add argument
    parser.add_argument('--opt',type=str, help='json配置文件/目录. eg. --opt "config_baidu.json"',default='config_baidu.json');
    parser.add_argument('--fun', type=int, help='功能(0-下载,1-批量生成json配置文件). eg. --fun 0', default=0);
    parser.add_argument('--save_path',type=str, help='下载保存目录. eg. --save_path "/temp/json"',default='%s/e/json_down'%(os.getenv('HOME')));
    parser.add_argument('--json_path',type=str, help='json配置保存目录. eg. --json_path "/temp/json"',default='%s/e/json_down'%(os.getenv('HOME')));
    parser.add_argument('--keys_file',type=str, help='搜索关键字，逗号隔开. eg. --keys_file "/temp/json/keys.txt"',default='%s/e/json_down/keys.txt'%(os.getenv('HOME')));
    parser.add_argument('--dbg', type=int, help='是否调试(1-调试,0-不调试). eg. --dbg 0', default=0);
    # Parse argument
    arg = parser.parse_args();
    #调试
    pdb.set_trace() if arg.dbg==1 else ''
    return arg


def main(arg):
    ''' 主函数
    '''
    if arg.fun==0:
        if os.path.exists(arg.opt):
            if os.path.isfile(arg.opt):
                obj=BaiduDownload()
                obj.downloads(arg.opt)
            else:
                obj=BaiduDownload()
                obj.downloads_path(arg.opt)
    elif arg.fun==1:
        obj=BaiduDownload()
        obj.create_json_config(arg.save_path,arg.json_path,arg.keys_file)
    else:
        print('%s not exists!'%(arg.opt))


if __name__=='__main__':
    arg=params()
    main(arg)

