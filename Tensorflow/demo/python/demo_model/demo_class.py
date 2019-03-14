#使用范例:
# obj=demo_class(1,2,3,4); #定义对象
# arg1=obj.arg1;           #获取属性
# obj.arg1=arg1;           #设置属性
# arg2=obj.get_arg2();     #获取属性
# obj.set_arg2(arg2);      #设置属性
# obj.print_arg();         #调用方法
# sType=type(obj);         #获取对象信息:类型
# isinstance(obj,demo_pclass);#类型判断
# attrs=dir(obj);          #对象的所有属性和方法
# attr=getattr(obj,"arg1");#获取属性
# setattr(obj,"arg1",3);   #设置属性
# hasattr(obj,"arg1");     #是否有属性
#
# 绑定方法
# from types import MethodType
# obj.fun1=MethodType(fun1,obj);
# obj.fun1(1,2);
#定义函数
def fun1(a,b):
    return a+b;

#父类
class demo_pclass(object):
    #限定属性
    __slots__=("arg1");

    #类似构造函数
    def __init__(self,arg1,arg2):
        self.arg1=arg1;  #参数arg1可以由外部直接访问
        self.__arg2=arg2;#前面加 __ 就变成私有成员,外部不能直接访问

    def get_arg2(self):
        return self.__arg2;

    def set_arg2(self,arg2):
        self.__arg2=arg2;

    def print_arg(self):
        print("arg1:%s,arg2:%s", %(self.arg1,self.__arg2));

#继承
class demo_class(demo_pclass):
    #类似构造函数
    def __init__(self,arg1,arg2,arg3,arg4):
        super(demo_class,demo_pclass).__init__(arg1,arg2);#调用父类构造函数

        self.arg3=arg3;  #参数arg1可以由外部直接访问
        self.__arg4=arg4;#前面加 __ 就变成私有成员,外部不能直接访问

    #@property装饰器就是负责把一个方法变成属性调用的
    @property
    def arg4(self):
        return self.__arg4;
    @arg4.setter
    def arg4(self,arg4):
        self.__arg4=arg4;

    def get_arg4(self):
        return self.__arg4;

    def set_arg4(self,arg4):
        self.__arg4=arg4;

    def print_arg(self):
        print("arg1:%s,arg2:%s,arg3:%s,arg4:%s", %(self.arg1,self.__arg2,self.arg3,self.__arg4));
