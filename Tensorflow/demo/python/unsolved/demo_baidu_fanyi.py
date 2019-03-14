#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''百度翻译
    url:
        https://fanyi.baidu.com/v2transapi
    header:
        Accept	:        */*
        Accept-Encoding	:        gzip, deflate, br
        Accept-Language	:        zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2
        Connection	:        keep-alive
        Content-Length	:        122
        Content-Type	:        application/x-www-form-urlencoded; charset=UTF-8
        Cookie	:        BAIDUID=0CD7138DDFACDA2DF4049E… delPer=0; PSINO=1; locale=zh
        Host	:        fanyi.baidu.com
        Referer	:        https://fanyi.baidu.com/
        User-Agent	:        Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:63.0) Gecko/20100101 Firefox/63.0
        X-Requested-With	:        XMLHttpRequest
    postData:
        from:	en
        query:	king
        sign:	612765.899756
        simple_means_flag:	3
        to:	zh
        token:	d6555f2c8dce35bc24baa6b6f57f6774
        transtype:	translang

    response = requests.post(url=url,data=posData,headers=headers)#模拟请求

    异常：
        eeor:997错误
    解决方法：暂无

'''

import requests #导入需要的包
import json

url='https://fanyi.baidu.com/v2transapi'
url='https://fanyi.baidu.com/basetrans'
headers={
    'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:63.0) Gecko/20100101 Firefox/63.0'
    ,'X-Requested-With':'XMLHttpRequest'
    ,'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8'
    ,'Host':'fanyi.baidu.com'
    ,'Referer':'https://fanyi.baidu.com/'
    }
posData={
    'from':'en'
    ,'to':'zh'
    ,'query':'king'
    ,'transtype':'translang'
    }
response=requests.post(url=url,data=posData,headers=headers)
json_data=json.loads(response.content.decode()) 
print(json_data)
