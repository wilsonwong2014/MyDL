#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################
#      函数定义范例        #
###########################

###########################
#空函数
#  空函数什么也不做
#使用范例：
#   my_fun_none();
#
def my_fun_none():
    pass

###########################
#有位置参数，无返回值函数
#使用范例：
#  my_fun1("str");
def my_fun1(val):
    print("------my_fun1------");
    print("val:",val);
    isinstance(val,int);
    isinstance(val,float);
    isinstance(val,str);
    isinstance(val,list);
    isinstance(val,tuple);
    isinstance(val,dict);
    isinstance(val,set);

###########################
#有位置参数，有单个返回值
#使用范例：
#  ret = my_fun2(1,2);
def my_fun2(val1,val2):
    print("------my_fun2------");
    print("val1:%s,val2:%s" %(val1,val2));
    ret = None;
    if isinstance(val1,(int,float)) and isinstance(val2,(int,float)):
        ret = val1 + val2;

    return ret

###########################
#有位置参数，有多个返回值
#使用范例:
#  my_fun3(1,2);
def my_fun3(val1,val2):
    print("------my_fun3------");
    print("val1:%s,val2:%s" %(val1,val2));
    ret1 = None;
    ret2 = None;
    if isinstance(val1,(int,float)) and isinstance(val2,(int,float)):
        ret1 = val1 + val2;    
        ret2 = val1 - val2;
    return ret1,ret2

###########################
#参数默认值
#使用范例:
#  ret = my_fun4(1,2);
def my_fun4(val1=1,val2=2):
    print("------my_fun4------");
    print("val1:%s,val2:%s" %(val1,val2));
    ret = None;
    if isinstance(val1,(int,float)) and isinstance(val2,(int,float)):
        ret = val1 + val2;
    return ret;



###########################
#可变参数
#使用范例:
#  my_fun5(1);
#  my_fun5(1,2);
#  l = [1,2,3];
#  my_fun5(*l);
def my_fun5(*args):
    print("------my_fun5------");
    for arg in args:
        print("arg:",arg);

###########################
#关键字参数
#使用范例:
#  my_fun6(key1="val1");
#  my_fun6(key1="val1",key2="val2");
#  d = {'key1':'val1','key2':'val2'};
#  my_fun6(**d);
def my_fun6(**args):
    print("------my_fun6------");
    for key,val in args.items():
        print("key:%s,val:%s" %(key,val));

###########################
#命名关键字参数
#使用范例:
#  #my_fun7(key1="val1");#有误，一定要所有命名关键字
#  my_fun7(key1="val1",key2="val2");
def my_fun7(*,key1,key2):
    print("------my_fun7------");
    print("key1:",key1);
    print("key2:",key2);

###########################
#参数组合（必选参数，默认参数，可变参数，关键字参数，命名关键字参数）
#  规则1:顺序必须为(必选参数,默认参数,可变参数/命名关键字参数,关键字参数)
#  规则2:可变参数与命名关键字参数不能同时使用
#  规则3:命名关键字参数必须全部出现
#使用范例:
#        my_fun8("val1");
#        my_fun8("val1","val2");
#        my_fun8("val1","val2","val3");
#        my_fun8("val1","val2","val3","val4");
#        my_fun8("val1","val2","val3","val4",key1="kv1",key2="kv2");
def my_fun8(val1,val2='val2',*args,**kv):
    print("------my_fun8------");
    print("val1:",val1);
    print("val2:",val2);
    for x in args:
        print("x:",x);
    for x in kv.items():
        print("kv:",x);

#使用范例:
#        my_fun9("val1",key1="kv1",key2="kv2");
#        my_fun9("val1","val2",key1="kv1",key2="kv2");
#        my_fun9("val1","val2",key1="kv1",key2="kv2",key3="kv3",key4="kv4");
def my_fun9(val1,val2='val2',*,key1,key2,**kv):
    print("------my_fun9------");
    print("val1:",val1);
    print("val2:",val2);
    print("key1:",key1);
    print("key2:",key2);
    for x in kv.items():
        print("kv:",x);

#############################
#递归函数
#使用范例:
#  ret = my_fun10(10);
def my_fun10(n):
    print("------my_fun10------");
    print("n:",n);
    if n==1:
        ret = 1;
    else:
        ret = n * my_fun10(n-1);
    return ret;

#############################
#尾递归防止堆栈溢出
#  规则：return 不能含有表达式，把计算放入函数参数完成.
#使用范例:
#  ret = my_fun11(10);
def my_fun11(n):
    print("------my_fun11------");
    print("n:",n);
    return my_fun11_iter(n,1);

def my_fun11_iter(n,p):
    print("------my_fun11_iter------");
    print("n:%d,p:%d" %(n,p));
    if n==1:
        return p;
    else:
        return my_fun11_iter(n-1,n*p);


#################################
#函数作为参数
#使用范例：
#   my_fun12(1,2,my_add);
def my_add(a,b):
    return a+b;

def my_fun12(a,b,fnAdd):
    return fnAdd(a,b);

#################################
#map/reduce
#使用范例：
#   my_map_test();
#   my_reduce_test();
def my_pow(x):
    return x*x;

def my_map_test():
    list1=range(10);
    list2=map(my_pow,list1);#[0,1,4,9,16,25,36,49,64,81]

def my_reduce_test():
    list1=range(10);
    #reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
    list2=reduce(my_add,list1);#[0+1+2+3+4+5+6+7+8+9]    

################################
#filter
#使用范例：
#    my_filter_test();
def my_isodd(x):
    return x%2==1;
def my_filter_test():
    list1=range(10);
    list2=filter(my_isodd,list1);#[1,3,5,7,9]

################################
#sorted
#使用范例：
def my_sorted_test():
    list1=sorted([36, 5, -12, 9, -21], key=abs);#=>[5, 9, -12, -21, 36]

###############################
#函数最为返回参数
#使用范例：
#  fn1 = fun_retfun(0,1,2);
#  fn2 = fun_retfun(1,1,2);
#  print(fn1());
#  print(fn2());
def fun_retfun(flag,*args):
    def fun1(*args):
        s=0;
        for x in args:
            s=s+x;
        return s;
    def fun2(*args):
        s=0;
        for x in args:
            s=s+x*x;
        return s;
    if flag==0:
        return fun1;
    elif flag==1:
        return fun2;

def my_fun13():
    fn1=fun_retfun(0);
    fn2=fun_retfun(1);
    print("fn1:%s,fn2:%s" %(fn1(1,2),fn2(1,2)));

################################
#匿名函数
def my_fun14():
    f=lambda x,y:x*y;
    print("f(2,3):", f(2,3));

################################
#偏函数:把函数的某些参数固定
def my_fun15():
    import functools
    int2 = functools.partial(int, base=2);
    int2('1000000');#=>64
    int2('1010101');#=>85

############## 测试 ###############
if __name__ == '__main__':
    try:
        my_fun_none();
        my_fun1("str");
        ret = my_fun2(1,2);
        [ret1,ret2]=my_fun3(1,2);
        ret = my_fun4(1,2);
        ret = my_fun4(val1=2);
        ret = my_fun4(val2=3);
        ret = my_fun4(val1=4,val2=5);
        ret = my_fun4(val2=6,val1=7);
        my_fun5(1);
        my_fun5(1,2);
        my_fun6(key1="val1");
        my_fun6(key1="val1",key2="val2");
        my_fun7(key1="val1",key2="val2");
        my_fun8("val1");
        my_fun8("val1","val2");
        my_fun8("val1","val2","val3");
        my_fun8("val1","val2","val3","val4");
        my_fun8("val1","val2","val3","val4",key1="kv1",key2="kv2");
        my_fun9("val1",key1="kv1",key2="kv2");
        my_fun9("val1","val2",key1="kv1",key2="kv2");
        my_fun9("val1","val2",key1="kv1",key2="kv2",key3="kv3",key4="kv4");
        ret = my_fun10(10);
        print("my_fun10(10)=>%d",ret);
        ret = my_fun11(10);
        ret=my_fun12(1,2,my_add);
        print("my_fun12(1,2,my_add);=>%s",ret);
        my_fun13();
        my_fun14();
    except rospy.ROSInterruptException:
        print("Error!");

