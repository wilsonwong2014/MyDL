#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################
#    字符串和编码 使用范例    #
##############################
#
# 使用范例:
#   python3 demo_str.py
#
########################################

import operator as op
import string
import locale

#字符串和编码
nCode1 = ord('A');  #字符=>编码,65
nChar1 = chr(65);   #编码=>字符,A
nCode2 = ord('中'); #字符=>编码,20013
nChar2 = chr(20013);#编码=>字符,中
sStr   = '\u4e2d\u6587';#字符的整数编码，中文
#字符串在内存以unicode存储
bytesVal1 = 'ABC'.encode('ascii'); #字符串编码为ascii=>b'ABC'
bytesVal2 = 'ABC'.encode('utf-8'); #字符串编码为utf-8=>b'ABC'
#bytesVal3 ='中文'.encode('ascii');#字符含有中文，编码为ascii报错
bytesVal4 = '中文'.encode('utf-8');#=>b'\xe4\xb8\xad\xe6\x96\x87'
#解码
sStr1 = b'ABC'.decode('ascii'); #=>'ABC'
sStr2 = b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8');#=>中文

#字符长度
nLen = len('abc');

#字符串格式化
str1 = 'str1:%s,%s' %('a','b');

###########################################
#Python 字符串操作方法大全
#1、去空格及特殊符号
#  s.strip(rm)       删除s字符串中开头、结尾处，位于 rm删除序列的字符
#  s.lstrip(rm)      删除s字符串中开头处，位于 rm删除序列的字符
#  s.rstrip(rm)      删除s字符串中结尾处，位于 rm删除序列的字符
#删除两边空格
s='  123  ';
s.strip(); #删除两边空格:=>'123'
s.lstrip();#删除左边空格:=>'123   '
s.rstrip();#删除右边空格:=>'   123'
ss='abcdefgab';
ss.strip('ab'); #删除字符串开头,结尾处的'ab':=>'cdefg'
ss.lstrip('ab');#删除字符串开头的'ab':=>'cdefgab'
ss.rstrip('ab');#删除字符串结尾处的'ab':=>'abcdefg'

#2、复制字符串
#strcpy(sStr1,sStr2)
sStr1 = 'strcpy'
sStr2 = sStr1
sStr1 = 'strcpy2'
print(sStr2); #=>'strcpy'

#3、连接字符串
#strcat(sStr1,sStr2)
sStr1 = 'strcat'
sStr2 = 'append'
sStr1 += sStr2
print(sStr1); #=>'strcatappend'

#4、查找字符
#strchr(sStr1,sStr2)
# < 0 为未找到
sStr1 = 'strchr'
sStr2 = 's'
nPos = sStr1.index(sStr2)
print(nPos); #=>0

#5、比较字符串
#strcmp(sStr1,sStr2)
sStr1 = 'strchr'
sStr2 = 'strch'
print(op.eq(sStr1,sStr2));#=>False

#6、扫描字符串是否包含指定的字符
#strspn(sStr1,sStr2)
sStr1 = '12345678'
sStr2 = '456'
sStr1.find(sStr2);#=>3

#7、字符串长度
#strlen(sStr1)
sStr1 = 'strlen'
print(len(sStr1)); #=>6

#8、将字符串中的大小写转换
s='aBcD'
s.lower();          #小写 :=>'abcd'
s.upper();          #大写 :=>'ABCD'
s.swapcase();       #大小写互换:=>'AbCd'
s.capitalize();     #首字母大写:=>'Abcd' 
ss='ab cd ef'
string.capwords(ss);#这是模块中的方法。它把S用split()函数分开，然后用capitalize()把首字母变成大写，最后用join()合并到一起 
                    #=>'Ab Cd Ef'

#9、追加指定长度的字符串
#strncat(sStr1,sStr2,n)
sStr1 = '12345'
sStr2 = 'abcdef'
n = 3
sStr1 += sStr2[0:n]
print(sStr1); #=>'12345abc'

#10、字符串指定长度比较
#strncmp(sStr1,sStr2,n)
sStr1 = '12345'
sStr2 = '123bc'
n = 3
print(op.eq(sStr1[0:n],sStr2[0:n]));#=>True

#11、复制指定长度的字符
#strncpy(sStr1,sStr2,n)
sStr1 = ''
sStr2 = '12345'
n = 3
sStr1 = sStr2[0:n]
print(sStr1); #=>'123' 

#12、将字符串前n个字符替换为指定的字符
#strnset(sStr1,ch,n)
sStr1 = '12345'
ch = 'r'
n = 3
sStr1 = n * ch + sStr1[3:]
print(sStr1); #=>'rrr45'

#13、扫描字符串
#strpbrk(sStr1,sStr2)
sStr1 = 'cekjgdklab'
sStr2 = 'gka'
nPos = -1
for c in sStr1:
    if c in sStr2:
        nPos = sStr1.index(c)
        break
print(nPos); #=>???

#14、翻转字符串
#strrev(sStr1)
sStr1 = 'abcdefg'
sStr1 = sStr1[::-1]
print(sStr1); #=>'gfedcba'

#15、查找字符串
#strstr(sStr1,sStr2)
sStr1 = 'abcdefg'
sStr2 = 'cde'
print(sStr1.find(sStr2)); #=>2

#16、分割字符串
#strtok(sStr1,sStr2)
sStr1 = 'ab,cde,fgh,ijk'
sStr2 = ','
sStr1 = sStr1[sStr1.find(sStr2) + 1:]
print(sStr1); #=>???
#或者
s = 'ab,cde,fgh,ijk'
print(s.split(',')); #=>['ab','cde','fgh','ijk']

#17、连接字符串
delimiter = ','
mylist = ['Brazil', 'Russia', 'India', 'China']
print(delimiter.join(mylist)); #=>'Brazil, Russia, India, China'

#18、PHP 中 addslashes 的实现
def addslashes(s):
    d = {'"':'\\"', "'":"\\'", "\0":"\\\0", "\\":"\\\\"}
    return ''.join(d.get(c, c) for c in s)

s = "John 'Johny' Doe (a.k.a. \"Super Joe\")\\\0"
print(s)
print(addslashes(s)); #=> ???

#19、只显示字母与数字
def OnlyCharNum(s,oth=''):
    s2 = s.lower();
    fomart = 'abcdefghijklmnopqrstuvwxyz0123456789'
    for c in s2:
        if not c in fomart:
            s = s.replace(c,'');
    return s;

print(OnlyCharNum("a000 aa-b")); #=>'a000aab'

#20、截取字符串
s='0123456789'
print(s[0:3])    #截取第一位到第三位的字符:=>'012'
print(s[:])      #截取字符串的全部字符:=>'0123456789'
print(s[6:])     #截取第七个字符到结尾:=>'6789'
print(s[:-3])    #截取从头开始到倒数第三个字符之前:=>'0123456'
print(s[2])      #截取第三个字符:=>'2'
print(s[-1])     #截取倒数第一个字符:=>'9'
print(s[::-1])   #创造一个与原字符串顺序相反的字符串:=>'9876543210'
print(s[-3:-1])  #截取倒数第三位与倒数第一位之前的字符:=>'78'
print(s[-3:])    #截取倒数第三位到结尾:=>'789'
print(s[:-5:-3]) #逆序截取，具体啥意思没搞明白？:=>96
                   #https://blog.csdn.net/win_turn/article/details/52998912

#21、字符串在输出时的对齐 
s='123'
width=5
fillchar='0'
s.ljust(width,fillchar)   #左对齐:=>'12300'
#输出width个字符，S左对齐，不足部分用fillchar填充，默认的为空格。 
s.rjust(width,fillchar)   #右对齐:=>'00123'
s.center(width, fillchar) #中间对齐:=>'01230'
s.zfill(width) #把S变成width长，并在右对齐，不足部分用0补足:=>'00123'

#22、字符串中的搜索和替换 
s='0123456789012'
substr='12'
start=0
end=10
oldstr='12'
newstr='ab'
count=2
chars='012'
s.find(substr, start, end)     #=>1
#返回S中出现substr的第一个字母的标号，如果S中没有substr则返回-1。start和end作用就相当于在S[start:end]中搜索 
s.index(substr, start, end)    #=>1 
#与find()相同，只是在S中没有substr时，会返回一个运行时错误 
s.rfind(substr, start, end)    #=>11
#返回S中最后出现的substr的第一个字母的标号，如果S中没有substr则返回-1，也就是说从右边算起的第一次出现的substr的首字母标号 
s.rindex(substr, start, end)   #=>11
s.count(substr, start, end) #计算substr在S中出现的次数 
s.replace(oldstr, newstr, count) #=>'0ab34567890ab'
#把S中的oldstar替换为newstr，count为替换次数。这是替换的通用形式，还有一些函数进行特殊字符的替换 
s.strip(chars)                   #=>'3456789'
#把S中前后chars中有的字符全部去掉，可以理解为把S前后chars替换为None 
s.lstrip(chars)                  #=>'3456789012'
s.rstrip(chars)                  #=>'0123456789'
s='\t0123456'
tabsize=8
s.expandtabs(tabsize)            #=>'        0123456'
#把S中的tab字符替换没空格，每个tab替换为tabsize个空格，默认是8个

#23、字符串的分割和组合 
s='ab,cd,ef,gh,ij'
sep=','
maxsplit=2
keepends=False
s.split(sep, maxsplit)         #=>['ab','cd','ef,gh,ij']
#以sep为分隔符，把S分成一个list。maxsplit表示分割的次数。默认的分割符为空白字符 
s.rsplit(sep, maxsplit)        #=>['ab,cd,ef','gh','ij'] 
s.splitlines(keepends)           #
#把S按照行分割符分为一个list，keepends是一个bool值，如果为真每行后而会保留行分割符。 
s=','
seq=['ab','cd','ef']
s.join(seq) #把seq代表的序列──字符串序列，用S连接起来:=>'ab,cd,ef'

#24、字符串的mapping，这一功能包含两个函数 
s_from='abcd'
s_to='1234'
table=str.maketrans(s_from, s_to)   #=>{97: 49, 98: 50, 99: 51, 100: 52}
#返回一个256个字符组成的翻译表，其中from中的字符被一一对应地转换成to，所以from和to必须是等长的。 
ss='abcdefg'
ss.translate(table) #=>'1234efg'
#ss.translate(table[,deletechars]) 
# 使用上面的函数产后的翻译表，把S进行翻译，并把deletechars中有的字符删掉。需要注意的是，如果S为unicode字符串，那么就不支持 deletechars参数，可以使用把某个字符翻译为None的方式实现相同的功能。此外还可以使用codecs模块的功能来创建更加功能强大的翻译表。

#25、字符串还有一对编码和解码的函数 
s='abc'
encoding='utf-8'
errors='strict'
ss=s.encode(encoding,errors) 
# 其中encoding可以有多种值，比如gb2312 gbk gb18030 bz2 zlib big5 bzse64等都支持。errors默认值为"strict"，意思是UnicodeError。可能的值还有'ignore', 'replace', 'xmlcharrefreplace', 'backslashreplace' 和所有的通过codecs.register_error注册的值。这一部分内容涉及codecs模块，不是特明白 
ss.decode(encoding,errors)

#26、字符串的测试、判断函数，这一类函数在string模块中没有，这些函数返回的都是bool值 
s='abcdefg'
prefix='abc'
start=0
end=10
s.startswith(prefix,start,end) 
#是否以prefix开头 
suffix='efg'
s.endswith(suffix,start,end) 
#以suffix结尾 
s.isalnum() 
#是否全是字母和数字，并至少有一个字符 
s.isalpha() #是否全是字母，并至少有一个字符 
s.isdigit() #是否全是数字，并至少有一个字符 
s.isspace() #是否全是空白字符，并至少有一个字符 
s.islower() #S中的字母是否全是小写 
s.isupper() #S中的字母是否便是大写 
s.istitle() #S是否是首字母大写的

#27、字符串类型转换函数，这几个函数只在string模块中有
base=10
locale.atoi(s,base) 
#base默认为10，如果为0,那么s就可以是012或0x23这种形式的字符串，如果是16那么s就只能是0x23或0X12这种形式的字符串 
string.atol(s,base) #转成long 
string.atof(s,base) #转成float

#    这里再强调一次，字符串对象是不可改变的，也就是说在python创建一个字符串后，你不能把这个字符中的某一部分改变。
#任何上面的函数改变了字符串后，都会返回一个新的字符串，原字串并没有变。其实这也是有变通的办法的，可以用S=list(S)
#这个函数把S变为由单个字符为成员的list，这样的话就可以使用S[3]='a'的方式改变值，然后再使用S=" ".join(S)还原成字符串


