#!/usr/bin/env python3
# -*- coding: UTF-8 -*- 

'''进度条显示
    https://blog.csdn.net/u013832707/article/details/73608504 
    使用范例：
    from ProcessBar import ShowProcess
    max_steps = 100
    process_bar = ShowProcess(max_steps,'head','tail', 'OK') 
    for i in range(max_steps): 
        #DoSomthing
        process_bar.show_process() 
        time.sleep(0.1)    
'''
import sys, time 
class ShowProcess(): 
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度 
    max_steps = 0 # 总共需要处理的次数 
    max_arrow = 50 #进度条的长度 
    infoDone = 'done' 

    # 初始化函数，需要知道总共的处理次数 
    def __init__(self, max_steps=100,infoHead='',infoTail='', infoDone = 'Done'):
        self.max_steps = max_steps 
        self.i = 0 
        self.infoHead=infoHead
        self.infoTail=infoTail
        self.infoDone = infoDone 

    # 显示函数，根据当前的处理进度i显示进度 
    # 效果为 headinfo [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] tailinfo
    def show_process(self, i=None,infoHead=None,infoTail=None): 
        if i is not None: 
            self.i = i 
        else: 
            self.i += 1 
        if infoHead is not None:
            self.infoHead=infoHead
        if infoTail is not None:
            self.infoTail=infoTail
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>' 
        num_line = self.max_arrow - num_arrow #计算显示多少个'-' 
        process_bar = self.infoHead + '[' + '>' * num_arrow + '-' * num_line + ']' + self.infoTail + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        #process_bar = self.infoHead + '[' + '>' * num_arrow + '-' * num_line + ']' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端 
        sys.stdout.flush() 
        if self.i >= self.max_steps: 
            self.close() 
    
    #进度条显示完成
    def close(self): 
        print('') 
        print(self.infoDone) 
        self.i = 0  

    #状态重置
    def reset(i,infoDone,infoHead,infoTail):
        self.i=i
        self.infoDone=infoDone
        self.infoHead=infoHead
        self.infoTail=infoTail

#----------------------------------
if __name__=='__main__': 
    max_steps = 100 
    process_bar = ShowProcess(max_steps,'head','tail', 'OK') 
    for i in range(max_steps): 
        #DoSomthing
        process_bar.show_process() 
        time.sleep(0.1)


