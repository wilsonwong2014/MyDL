#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
SQLLite数据库操作范例
SQLite 教程
   http://www.runoob.com/sqlite/sqlite-tutorial.html
常用命令
   启用sqlite3
     $sqlite3 db.db
     sqlite>
   帮助信息
     sqlite>.help
   列举当前数据库
     sqlite>.database
   列举当前数据表
     sqlite>.tables
   显示数据表结构
     sqlite>.schema tablename
   显示表头
     sqlite>.headers on
   按列对齐输出结果
     sqlite>.mode column
   显示当前配置信息
     sqlite>.show
   显示主表
     sqlite>select * from sqlite_master
   退出sqlite
     sqlite>.quit

Cursor常用方法:
  1. c.move(int offset);                  //以当前位置为参考,移动到指定行  
  2. c.moveToFirst();                     //移动到第一行  
  3. c.moveToLast();                      //移动到最后一行  
  4. c.moveToPosition(int position);      //移动到指定行  
  5. c.moveToPrevious();                  //移动到前一行  
  6. c.moveToNext();                      //移动到下一行  
  7. c.isFirst();                         //是否指向第一条 
  8. c.isLast();                          //是否指向最后一条  
  9. c.isBeforeFirst();                   //是否指向第一条之前  
  10. c.isAfterLast();                    //是否指向最后一条之后  
  11. c.isNull(int columnIndex);          //指定列是否为空(列基数为0)  
  12. c.isClosed();                       //游标是否已关闭  
  13. c.getCount();                       //总数据项数  
  14. c.getPosition();                    //返回当前游标所指向的行数  
  15. c.getColumnIndex(String columnName);//返回某列名对应的列索引值  
  16. c.getString(int columnIndex);       //返回当前行指定列的值  
'''

import os;
import sys;
############# 调试 begin ###############
argc = len(sys.argv);
import pdb       
if argc>1 and sys.argv[1]=='dbg':    
    pdb.set_trace(); #调试
############# 调试 end   ###############


import sqlite3

#链接数据库
conn = sqlite3.connect('test.db')
print("Opened database successfully")

cursor=conn.cursor()

#判断数据表是否存在
#select * from sqlite_master where type = 'table' and name='tbl21';
cur = cursor.execute("SELECT *  from sqlite_master where type='table' and name='tbl1' ")
conn.commit()
if cur.fetchone() is None:
    #创建数据表
    conn.execute("create table tbl1(field1 int,field2 text) ")
    conn.commit()

#检测数据表字段
cur=cursor.execute("select * from sqlite_master where name='tbl1' and sql like '%field3%' ")
conn.commit()
if cur.fetchone() is None:
   #添加字段
   conn.execute("ALTER TABLE tbl1 ADD COLUMN field3 real ")
   conn.commit()

#添加记录
for i in range(10):
    sql="insert into tbl1(field1,field2,field3) values(%d,'%s',%f) " %(i,'val',i*10)
    conn.execute(sql)
conn.commit()

#删除记录
conn.execute("delete from tbl1 where field1=1")
conn.commit()

#修改记录
conn.execute("update tbl1 set field3=0 where field1=3")
conn.commit()

#查询
cursor=conn.execute("select * from tbl1")
conn.commit()
for row in cursor:
    print("field1:%s,field2:%s,field3:%s" %(row[0],row[1],row[2]))

#关闭链接
conn.close()

