#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
 时间模块time使用范例

在Python中，与时间处理有关的模块就包括：time，datetime以及calendar。这篇文章，主要讲解time模块。
在开始之前，首先要说明这几点：
1.在Python中，通常有这几种方式来表示时间：1）时间戳 2）格式化的时间字符串 3）元组（struct_time）
  共九个元素。由于Python的time模块实现主要调用C库，所以各个平台可能有所不同。
2.UTC（Coordinated Universal Time，世界协调时）亦即格林威治天文时间，世界标准时间。在中国为UTC+8。
  DST（Daylight Saving Time）即夏令时。
3.时间戳（timestamp）的方式：通常来说，时间戳表示的是从1970年1月1日00:00:00开始按秒计算的偏移量。
  我们运行“type(time.time())”，返回的是float类型。返回时间戳方式的函数主要有time()，clock()等。
4.元组（struct_time）方式：struct_time元组共有9个元素，返回struct_time的函数主要有gmtime()，
  localtime()，strptime()。下面列出这种方式元组中的几个元素：
  
  struct_time
  ----------------------------------------------------
  索引（Index）	属性（Attribute）	  值（Values）
  0             tm_year（年）             比如2011 
  1             tm_mon（月）              1 - 12
  2             tm_mday（日）             1 - 31
  3             tm_hour（时）             0 - 23
  4             tm_min（分）              0 - 59
  5             tm_sec（秒）              0 - 61
  6             tm_wday（weekday）        0 - 6（0表示周日）
  7             tm_yday（一年中的第几天）  1 - 366
  8             tm_isdst（是否是夏令时）   默认为-1

  
  time.strftime(format[, t])
  -------------------------------------------------
  格式  含义                                                                                   备注
  %a    本地（locale）简化星期名称                                                              .
  %A    本地完整星期名称                                                                        .
  %b    本地简化月份名称                                                                        .
  %B    本地完整月份名称                                                                        .
  %c    本地相应的日期和时间表示                                                                 .   
  %d    一个月中的第几天（01 - 31）                                                              .
  %H    一天中的第几个小时（24小时制，00 - 23）                                                   .
  %I    第几个小时（12小时制，01 - 12）                                                          .
  %j    一年中的第几天（001 - 366）                                                              .
  %m    月份（01 - 12）                                                                          .
  %M    分钟数（00 - 59）                                                                        .
  %p    本地am或者pm的相应符	                                                                一
  %S    秒（01 - 61）	                                                                        二
  %U    一年中的星期数。（00 - 53星期天是一个星期的开始。）第一个星期天之前的所有天数都放在第0周。	三
  %w    一个星期中的第几天（0 - 6，0是星期天）	                                                三
  %W    和%U基本相同，不同的是%W以星期一为一个星期的开始。                                         .
  %x    本地相应日期                                                                             .
  %X    本地相应时间                                                                             .
  %y    去掉世纪的年份（00 - 99）                                                                 .
  %Y    完整的年份                                                                               .
  %Z    时区的名字（如果不存在为空字符）                                                           .
  %%	‘%’字符                                                                                   .
  ----------------------------------
  备注：
    1.“%p”只有与“%I”配合使用才有效果。
    2.文档中强调确实是0 - 61，而不是59，闰年秒占两秒（汗一个）。
    3.当使用strptime()函数时，只有当在这年中的周数和天数被确定的时候%U和%W才会被计算。
'''

import time

#返回当前时间的时间戳=>float
cur_time=time.time()  ##1304576839.0
print('cur_time:',cur_time)

#返回格林威治时间=>struct_time
#time.struct_time(tm_year=2011, tm_mon=5, tm_mday=5, tm_hour=16, tm_min=37, tm_sec=6, tm_wday=3, tm_yday=125, tm_isdst=-1)
utc_time1=time.gmtime() 
utc_time2=time.gmtime(cur_time)
print('utc_time1:',utc_time1)
print('utc_time2:',utc_time2)

#返回当地时间=>struct_time
#time.struct_time(tm_year=2011, tm_mon=5, tm_mday=5, tm_hour=16, tm_min=37, tm_sec=6, tm_wday=3, tm_yday=125, tm_isdst=-1)
local_time1=time.localtime()
local_time2=time.localtime(cur_time)
print('local_time1:',local_time1)
print('local_time2:',local_time2)

#将struct_time转化为时间戳=>float
time1=time.mktime(local_time1)  #1304576839.0
print('time1:',time1)

#sleep
time.sleep(2)

#时间格式化输出
str_time=time.strftime('%Y-%m-%d %H:%M:%S',local_time1) #=>'2018-09-10 09:01:30'
print('str_time:',str_time)

#格式化字符串转化为struct_time
#time.struct_time(tm_year=2011, tm_mon=5, tm_mday=5, tm_hour=16, tm_min=37, tm_sec=6, tm_wday=3, tm_yday=125, tm_isdst=-1)
t_time=time.strptime(str_time,'%Y-%m-%d %H:%M:%S')
print('t_time:',t_time)


