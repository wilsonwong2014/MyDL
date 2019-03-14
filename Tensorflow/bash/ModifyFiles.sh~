#!/bin/sh

########################################
#模块名称：文本文件内容修改
#模块功能：修改文本文件内容
#   把含有OldName的内容替换为sNewName内容
#   操作：
#      1.文本文件搜索替换:把含有OldContent替换为NewContent
#使用范例：
#   $bash ModifyFiles.sh ~/MyPrjs/ws_catkin/src/demo_pkg1 sOldContent sNewContent
#
######################################

######################################
#递归遍历文件夹
# 输入参数：
#   $1 --- 当前目录
#   $2 --- 旧文本
#   $3 --- 新文本
travFolder()
{ 
    #echo "travFolder"
    sPath=$1
    sOldContent=$2
    sNewContent=$3
    echo "travFolder $sPath $sOldContent $sNewContent"
    flist=`ls $sPath`
    cd $1
    #echo $flist
    for f in $flist
    do
        if test -d $f
        then
            echo "---dir:$f ---"
            travFolder $f $sOldContent $sNewContent
        else
            echo "---file:$f ---"
            #文件内容修改
            echo "sed -i \"s/$sOldContent/$sNewContent/g\" $f"
            sed -i "s/$sOldContent/$sNewContent/g" $f
        fi
    done
    cd ../ 
}

######################################
#参数个数
nArgNum=$#
if [[ $nArgNum -ne 3 ]]
then
  echo "error:not enough argument! Argument Number must 3,but current ArgNum:$nArgNum"
  echo "usage:bash ModifyFiles.sh /Path sOldContent sNewContent"
  exit 1
fi

#搜索目录
sPath=$1

#旧文本
sOldContent=$2

#新文本
sNewContent=$3

#判断目录是否存在
if [ ! -d "$sPath" ]; then
  echo "error!"
  echo "Path:$sPath not exist!"
  exit 1
fi

#遍历文件夹
travFolder $sPath $sOldContent $sNewContent

