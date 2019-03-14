函数，类，模块使用例程

模块：
　　一个文件就是一个模块，文件名就是模块名；模块之间的名称(变量,函数)不冲突；
包:
    一个目录就是一个功能包,目录下必须含有 __init__.py文件.

Python模块搜索路径代码详解
    https://www.jb51.net/article/133925.htm
    动态增加路径
    通过 sys 模块的 append() 方法在 Python 环境中增加搜索路径：
        >>> import sys
        >>> sys.path.append('/home/wang/workspace')
    修改 PYTHONPATH 变量
    打开并编辑 bashrc：
        $ vim ~/.bashrc
        将以下内容附加到文件末尾：
            export PYTHONPATH=$PYTHONPATH:/home/wang/workspace
        $ source ~/.bashrc

    增加 .pth 文件
    在 /usr/local/lib/python3.5/dist-packages 下添加一个扩展名为 .pth 的配置文件（例如：extras.pth），内容为要添加的路径：
        /home/wang/workspace

