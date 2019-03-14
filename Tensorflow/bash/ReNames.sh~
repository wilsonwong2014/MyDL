#!/bin/sh

########################################
#模块名称：递归修改文件夹/文件名
#模块功能：
#   递归遍历文件夹
#   操作：
#      1.文件夹改名:把含有OldName替换为NewName
#      2.文件名改名:把含有OldName替换为NewName
#使用范例：
#   $bash ReNames.sh ~/MyPrjs/ws_catkin/src/demo_pkg1 demo_pkg1 demo_pkg2
#
#   #源功能包:<demo_pkg1>
#   /demo_pkg1/demo_pkg1_node1.cpp
#
#   #修改后功能包:<demo_pkg2>
#   /demo_pkg2/demo_pkg2_node1.cpp
#
######################################

######################################
#递归遍历文件夹
# 输入参数：
#   $1 --- 当前目录
#   $2 --- 旧名称
#   $3 --- 新名称
travFolder()
{ 
    #echo "travFolder"
    sPath=$1
    sOldName=$2
    sNewName=$3
    echo "travFolder $sPath $sOldName $sNewName"
    flist=`ls $sPath`
    cd $1
    #echo $flist
    for f in $flist
    do
        if test -d $f
        then
            echo "---dir:$f ---"
            travFolder $f $sOldName $sNewName
            #目录改名
            sOldDirName=$f
            sNewDirName=${sOldDirName/$sOldName/$sNewName}
            if [ "$sOldDirName" != "$sNewDirName" ]
            then
               echo "mv $sOldDirName $sNewDirName"
               mv $sOldDirName $sNewDirName
            fi
        else
            echo "---file:$f ---"
            #文件改名
            sOldFileName=$f
            sNewFileName=${sOldFileName/$sOldName/$sNewName}
            echo "sOldFileName:$sOldFileName"
            echo "sNewFileName:$sNewFileName"
            if [ "$sOldFileName" != "$sNewFileName" ]
            then
               echo "mv $sOldFileName $sNewFileName"
               mv $sOldFileName $sNewFileName
            fi
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
  echo "usage:bash ReNames.sh /Path sOldName sNewName"
  exit 1
fi

#搜索目录
sPath=$1

#旧名称
sOldName=$2

#新名称
sNewName=$3

#判断目录是否存在
if [ ! -d "$sPath" ]; then
  echo "error!"
  echo "Path:$sPath not exist!"
  exit 1
fi

#目录改名
sOldDirName=$sPath
sNewDirName=${sOldDirName/$sOldName/$sNewName}
if [ "$sOldDirName" != "$sNewDirName" ]
then
   echo "mv $sOldDirName $sNewDirName"
   mv $sOldDirName $sNewDirName
fi
sPath=$sNewDirName

#遍历文件夹
travFolder $sPath $sOldName $sNewName


