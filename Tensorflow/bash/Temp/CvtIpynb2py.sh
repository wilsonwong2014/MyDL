#!/bin/sh

########################################
#模块名称：批量转换ipynb=>py
#模块功能：
#   递归遍历文件夹,转换*.ipynb文件为*.py文件.
#   脚本参数
#      $1 --- 遍历文件夹
#使用范例：
#   $bash CvtIpynb2py.sh ~/Data2/ipynb
#
######################################

######################################
#递归遍历文件夹
# 输入参数：
#   $1 --- 当前目录
travFolder()
{ 
    #echo "travFolder"
    sPath=$1
    echo "travFolder $sPath" 
    flist=`ls $sPath`
    cd $1
    #echo $flist
    for f in $flist
    do
        if test -d $f
        then
            echo "---dir:$f ---"
            travFolder $f 
        else
            echo "---file-1:$f ---"
            #文件改名
            sFile=$f
            #提取扩展名
            sExt=${sFile##*.}
            if [ "$sExt" == "ipynb" ]
            then
               echo "---file-2:$f ---"
               echo "jupyter nbconvert --to script ${PWD}/${sFile}"
               #jupyter nbconvert --to script ${PWD}/${sFile}
            fi
        fi
    done
    cd ../ 
}

######################################
#参数个数
nArgNum=$#
if [[ $nArgNum -ne 1 ]]
then
  echo "error:not enough argument! Argument Number must 1,but current ArgNum:$nArgNum"
  echo "usage:bash CvtIpynb2py.py /Path"
  exit 1
fi

#遍历目录
sPath=$1
#遍历文件夹
travFolder $sPath

