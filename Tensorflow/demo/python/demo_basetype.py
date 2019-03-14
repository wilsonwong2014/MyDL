#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################
#     基本数据类型演示     #
###########################

###########################
#整数：十进制
nVal10 = 100;

###########################
#整数：十六进制
nVal16 = 0x00FF;

###########################
#浮点数
fVal1 = 1.123;
fVal2 = 1.2e3;

###########################
#字符串
sVal = "string";

###########################
#布尔类型
bVal1 = True;
bVal2 = False;

###########################
#空值
noneVal = None;

###########################
#list ---- 可变向量表
listVal = ['a','b','c'];
#list:追加
listVal.append('d');
#list:弹出
sVal = listVal.pop();
#list:插入
listVal.insert(1,'e');
#list:移除
sVal = listVal.pop(1);
#list:访问
sVal = listVal[0]; #第一个
sVal = listVal[-1];#倒数第一个
#list:遍历
nLen = len(listVal);
for x in listVal:
    print(x);

###########################
#tuple --- 不可变向量表
tupleVal = ("1","2","3");
sVal = tupleVal[0]; #第一个
sVal = tupleVal[-1];#倒数第一个
for x in tupleVal:
    print(x);

###########################
#dict --- 字典
dictVal = {"key1":"val1","key2":"val2","key3":"val3"};
#dict:set
dictVal["key4"] = "val4";
#dict:get
sVal = dictVal.get("key3");
#dict:pop
sVal = dictVal.pop("key2");
#dict:访问
for key in dictVal:
    print("key:%s,val:%s" %(key,dictVal[key]));
for key in dictVal.keys():
    print("key:%s,val:%s" %(key,dictVal[key]));
for val in dictVal.values():
    print("val:",val);
for kv in dictVal.items():
    print(kv);
for key,val in dictVal.items():
    print("key:%s,val:%s" %(key,val));

###########################
#set --- 集合
setVal = set(["a","b","c"]);
#set:添加
setVal.add("d");
#set:移除
setVal.remove("b");
#set:判断
bExist = "a" in setVal;
#set:遍历
for x in setVal:
    print(x);


