#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#一、内存、CPU、硬盘、登录用户、进程等的一些基本信息：
import os
import sys
import psutil
import time
import datetime

import socket
import uuid
 
"""
获取系统基本信息
"""
 
EXPAND = 1024 * 1024
 
def mems():
    ''' 获取系统内存使用情况 '''
    mem = psutil.virtual_memory()
    mem_str = " 内存状态如下:\n"
    mem_str += "   系统的内存容量为: " + str(mem.total / EXPAND) + " MB\n"
    mem_str += "   系统的内存已使用容量为: " + str(mem.used / EXPAND) + " MB\n"
    mem_str += "   系统可用的内存容量为: " + str(mem.total / EXPAND - mem.used / (1024 * 1024)) + " MB\n"
    mem_str += "   内存的buffer容量为: " + str(mem.buffers / EXPAND) + " MB\n"
    mem_str += "   内存的cache容量为:" + str(mem.cached / EXPAND) + " MB\n"
    return mem_str
 
 
def cpus():
    ''' 获取cpu的相关信息 '''
    cpu_str = " CPU状态如下:\n"
    cpu_status = psutil.cpu_times()
    cpu_str += "   user = " + str(cpu_status.user) + "\n"
    cpu_str += "   nice = " + str(cpu_status.nice) + "\n"
    cpu_str += "   system = " + str(cpu_status.system) + "\n"
    cpu_str += "   idle = " + str(cpu_status.idle) + "\n"
    cpu_str += "   iowait = " + str(cpu_status.iowait) + "\n"
    cpu_str += "   irq = " + str(cpu_status.irq) + "\n"
    cpu_str += "   softirq = " + str(cpu_status.softirq) + "\n"
    cpu_str += "   steal = " + str(cpu_status.steal) + "\n"
    cpu_str += "   guest = " + str(cpu_status.guest) + "\n"
    return cpu_str
 
 
def disks():
    ''' 查看硬盘基本信息 '''
    ''' psutil.disk_partitions()    获取磁盘的完整信息
        psutil.disk_usage('/')      获得分区的使用情况,这边以根分区为例
        psutil.disk_io_counters()   获取磁盘总的io个数
        perdisk 默认为False
        psutil.disk_io_counters(perdisk=True)   perdisk为True 返回单个分区的io个数
    '''
    disk_str = " 硬盘信息如下:\n"
    disk_status = psutil.disk_partitions()
    for item in disk_status:
        disk_str += str(item) + "\n"
        p = item.device
        disk = psutil.disk_usage(p)
        disk_str += p+"盘容量为: " + str(disk.total / EXPAND) + " MB\n"
        disk_str += p+"盘已使用容量为: " + str(disk.used / EXPAND) + " MB\n"
        disk_str += p+"盘可用的内存容量为: " + str(disk.free / EXPAND) + " MB\n"
    return disk_str
 
 
def users():
    ''' 查看当前登录的用户信息 '''
    user_str = " 登录用户信息如下:\n "
    user_status = psutil.users()
    for item in user_status:
        user_str += str(item) + "\n"
    return user_str
 
def process():
    ''' 查看进程信息 '''
    pids = psutil.pids()
    proces = []
    for pid in pids:
        p = psutil.Process(pid)
        jctime = str(datetime.datetime.fromtimestamp(p.create_time()))[:19]
        p_info = [
            p.name(),       # 进程的名字
            p.exe(),        # 进程bin文件位置
            p.cwd(),        # 进程的工作目录的绝对路径
            p.status(),     # 进程的状态
            jctime,         # 进程的创建时间
            p.uids(),       # 进程的uid信息
            p.gids(),       # 进程的gid信息
            p.cpu_times(),  # cup时间信息
            p.memory_info(),# 进程内存的利用率
            p.io_counters() # 进程的io读写信息
        ]
        proces.append(p_info)
    return proces
 
#二、获取网络、网卡的信息：
def mac_name_ip():
    ''' 获得Mac地址、计算机名、IP地址 '''
    mac = uuid.UUID(int = uuid.getnode()).hex[-12:]	# Mac地址
    name = socket.getfqdn(socket.gethostname())	# 计算机名称
    addr = socket.gethostbyname(name)		# IP地址
    print('获得Mac地址、计算机名、IP地址')
    print('Mac地址:',mac)
    print('计算机名:',name)
    print('IP地址:',addr)
    return mac,name,addr
 
 
def net_all():
    ''' 获取网络总的IO信息 '''
    n = psutil.net_io_counters()
    ns = [
        n.bytes_sent,   # 发送字节数
        n.bytes_recv,   # 接受字节数
        n.packets_sent, # 发送数据包数
        n.packets_recv, # 接收数据包数
        ]
    print('获取网络总的IO信息')
    print('发送字节数:',n.bytes_sent)
    print('接受字节数:',n.bytes_recv)
    print('发送数据包数:',n.packets_sent)
    print('接收数据包数:',n.packets_recv)
    return ns

def net_line():
    ''' 获取每个网络接口的IO信息 '''
    n = psutil.net_io_counters(pernic=True)
    ns = {}
    for i in n:
        ns[i] = [
            n[i].bytes_sent,    # 发送字节数
            n[i].bytes_recv,    # 接受字节数
            n[i].packets_sent,  # 发送数据包数
            n[i].packets_recv,  # 接收数据包数
            ]

    print('获取每个网络接口的IO信息')
    for i in ns:
        print('发送字节数:',ns[i][0])
        print('接受字节数:',ns[i][1])
        print('发送数据包数:',ns[i][2])
        print('接收数据包数:',ns[i][3])

    return ns
 
def net_card():
    ''' 网卡信息
        返回：{网卡名：[IP地址64，IP地址32，Mac地址]，...}
    '''
    n = psutil.net_if_addrs()
    ns = {}
    for i in n:
        ns[i] = []
        for j in n[i]:
            ns[i].append(j.address)
    return ns

def net_cart_status():
    ''' 网卡状态 '''
    n = psutil.net_if_stats()
    return n
 
def network_connect():
    ''' 网路连接信息 '''
    n = psutil.net_connections()
    return n

#三、系统内存与CUP使用率：
def system_rate():
    ''' 内存与CPU使用率 '''
    #获取当前运行的pid
    p1=psutil.Process(os.getpid()) 
 
    #本机内存的占用率
    print ('内存占用率： '+str(psutil.virtual_memory().percent)+'%')

    #本机cpu的总占用率
    print ('CPU占用率： '+str(psutil.cpu_percent(0))+'%')

    #该进程所占cpu的使用率
    print ("该进程CPU占用率: "+str(p1.cpu_percent(None))+"%")
 
    #该进程所占内存占用率
    print ("该进程内存占用率: "+str(p1.memory_percent())+"%")
 

if __name__ == '__main__':
    print(mems())   # 内存
    print(cpus())   # CPU
    print(disks())  # 硬盘
    print(users())  # 登录用户
    proces = process()
    print(proces[0])

    system_rate()
    mac_name_ip()
    net_all()
    net_line()

