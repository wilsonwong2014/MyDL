#!/usr/bin/env sh

# Tensorflow-master docker启动脚本
# Tensorflow项目源码地址:
#    http://github.com/tensorflow/tensorflow
# Models项目源码地址
#    http://github.com/tensorflow/models   
#

#=========================
#常用命令
# 下载官方docker镜像
#   docker pull tensorflow/tensorflow:latest-devel-gpu-py3
# 启动镜像
#   docker run -it --rm --runtime=nvidia  tensorflow/tensorflow:latest-devel-gpu-py3
# 停止容器
#   docker stop container_id
# 启动容器
#   docker start container_id
# 进入容器交互界面
#   docker attch container_id
# 查看容器列表
#   docker ps -a
# 镜像保存:可以到任何地方加载
#   docker save image_name -o image.tar
# 镜像加载
#   docker load -i image.tar  
# 导出容器快照:保存所有file system
#   docker export 7691a814370e > ubuntu.tar 
# 导入容器快照
#   cat ubuntu.tar | docker import - test/ubuntu:v1.0
#===========================

#运行镜像: /tensorflow/models/research/deeplab/deeplab_demo.ipynb   OK!
#docker run -it --rm --runtime=nvidia -p 8888:8888 \
#    --mount type=bind,source=~/work/MyDL/Tensorflow/docker/tf_master,target=/root/work \
#    --mount type=bind,source=~/work/MyDL/books_examples/models-master,target=/tensorflow/models \
#    tensorflow/tensorflow:latest-devel-gpu-py3

#启动参数
#image=tensorflow/tensorflow:latest-devel-gpu-py3     #镜像名
if [ $# -gt 0 ]; then
image=$1
else
image=tensorflow/tensorflow:latest-devel-gpu-py3     #镜像名
fi

models_master=~/work/MyDL/books_examples/models-master   # models_master 路径
work_dir=~/work/MyDL/Tensorflow/docker/tf_master         # 工作目录
data_dir=~/work/data                                     # 数据目录
temp_dir=~/work/temp                                     # 临时目录
platform_dir=~/work/platform                             # 平台相关
#rm_exit="--rm"  #推出是否清除
nv="--runtime=nvidia"

if [ $image == "help" ];then
    echo "docker run -it ${rm_exit} --runtime=nvidia -p 8888:8888 \\"
    echo "    --mount type=bind,source=$work_dir,target=/root/work \\"
    echo "    --mount type=bind,source=$models_master,target=/tensorflow/models \\"
    echo "    tensorflow/tensorflow:latest-devel-gpu-py3"
else
    #运行镜像
    #docker run -it ${rm_exit} --runtime=nvidia -p 8888:8888 \
    #    --mount type=bind,source=$work_dir,target=/root/work \
    #    --mount type=bind,source=$models_master,target=/tensorflow/models \
    #    ${image} bash

    docker run -it ${rm_exit} --runtime=nvidia -p 8889:8888 \
        --mount type=bind,source=$work_dir,target=/root/work \
        --mount type=bind,source=$data_dir,target=/root/data \
        --mount type=bind,source=$temp_dir,target=/root/temp \
        --mount type=bind,source=$platform_dir,target=/root/platform \
        --mount type=bind,source=$models_master,target=/tensorflow/models \
        ${image} bash
fi

