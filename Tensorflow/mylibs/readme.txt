自定义模块，封装通用功能代码
为了方便调用模块，添加模块搜索目录（https://www.jb51.net/article/133925.htm）
    在 /usr/local/lib/python3.5/dist-packages 下添加一个扩展名为 .pth 的配置文件（例如：mylibs.pth），内容为要添加的路径：
        /home/hjw/MyDL/Tensorflow

每个模块的根目录必须有一个初始化文件以标识本目录为模块
    __init__.py


