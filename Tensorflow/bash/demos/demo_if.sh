#!/usr/bin/env bash
#Shell语言————if条件判断之组合判断（与、或、非）和多分支if语句
#   https://blog.51cto.com/wuyelan/1530277
#linux bash Shell特殊变量：Shell $0, $#, $*, $@, $?, $$和命令行参数
#   https://www.cnblogs.com/chjbbs/p/6393805.html
#   $? 上一个命令的退出状态
#   特殊变量列表
#   变量 含义
#   $0 当前脚本的文件名
#   $n 传递给脚本或函数的参数。n 是一个数字，表示第几个参数。例如，第一个参数是$1，第二个参数是$2。
#   $# 传递给脚本或函数的参数个数。
#   $* 传递给脚本或函数的所有参数。
#   $@ 传递给脚本或函数的所有参数。被双引号(" ")包含时，与 $* 稍有不同，下面将会讲到。
#   $? 上个命令的退出状态，或函数的返回值。
#   $$ 当前Shell进程ID。对于 Shell 脚本，就是这些脚本所在的进程ID。
#Linux命令之for - Bash中的For循环
#   https://blog.csdn.net/astraylinux/article/details/7016212
#shell中的while循环实例
#   https://blog.csdn.net/wdz306ling/article/details/79602739

#一、组合条件判断
#组合条件测试是指可以将多个条件组合起来进行判断，条件和条件之间有逻辑关系。
#例如判断一个数是否大于3，并且小于9，这里大于3是一个条件，小于9也是一个条件，这两个条件必须同时满足。
#同时满足即为逻辑关系。通常逻辑关系有以下几种：
#   与：-a，当指定多个条件时，默认为与关系
#   或：-o
#   非：!，这是个单目操作符

#如判断一个UID是否大于1，且小于499的写法如下：
#[root@localhost tutor]# Uid=300
#[root@localhost tutor]# [ $Uid -ge 1 ]
#[root@localhost tutor]# echo "$?"
# 0
#[root@localhost tutor]# [ $Uid -le 499 ]
#[root@localhost tutor]# echo $?
# 0
#[root@localhost tutor]# [ $Uid -ge 1 -a $Uid -le 499 ]

# 使用-a表示两个与关系的条件，必须同时满足
#[root@localhost tutor]# echo "$?"
# 0
#[root@localhost tutor]# Uid=3000
#[root@localhost tutor]# [ $Uid -ge 1 -a $Uid -le 499 ]
#[root@localhost tutor]# echo "$?"
# 1

#如判断一个UID是否等于0，或者大于的写法如下：
#[root@localhost tutor]# Uid=300
#[root@localhost tutor]# [ $Uid -eq 0 -o $Uid -ge 500 ]

## 使用-o表示两个或关系的条件，只需要满足其一即可
#[root@localhost tutor]# echo "$?"
# 1
#[root@localhost tutor]# Uid=3000
#[root@localhost tutor]# [ $Uid -eq 0 -o $Uid -ge 500 ]
#[root@localhost tutor]# echo "$?"
# 0

#判断一个UID是否不等于0，写法如下：
#[root@localhost tutor]# Uid=0
#[root@localhost tutor]# [ ! $Uid -eq 0 ]

# 使用! 表示取反，这里对Uid等于0的判断结果进行取反，即为Uid不等于0
#[root@localhost tutor]# echo "$?"
# 1
#[root@localhost tutor]# [ $Uid -ne 0 ]

# 这里判断Uid是否不等于0
#[root@localhost tutor]# echo "$?"
# 1


#二、多分支if语句
#前文中涉及到的条件判断语句，只有单分支和双分支的情况，事实上bash也支持多分支的条件判断，多分支的if语句是对双分支if语句的扩展。多分支if语句提供多个if条件，但仅执行其中一个语句，其语法格式为：
#if 条件1; then
#   语句1
#   语句2
#   ...
#elif 条件2; then
#   语句1
#   语句2
#   ...
#elif 条件3; then
#   语句1
#   语句2
#   ...
#else
#   语句1
#   语句2
#   ...
#fi


