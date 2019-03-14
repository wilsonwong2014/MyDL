#!/bin/bash

getdir()
{
    for element in `ls $1`
    do  
        dir_or_file=$1"/"$element
        if [ -d $dir_or_file ]
        then             
            if [ -f "$dir_or_file/Makefile" ];
            then
              echo "$dir_or_file/Makefile" 
              #echo "$dir_or_file/Makefile exist!"
              make -f "$dir_or_file/Makefile" clean
            fi

            getdir $dir_or_file
        #else
            #echo "file:$dir_or_file"
            #echo "elm :$element"
            #if [ "$element" = "Makefile" ]
            #then
            #   echo " $element is Makefile"
            #fi
        fi  
    done
}
#root_dir="."
#getdir $root_dir
getdir $1

#以下命令均不包含"."，".."目录，以及"."开头的隐藏文件，如需包含，ll 需要加上 -a参数
#当前目录下文件个数，不包含子目录
#ll |grep "^-"|wc -l
#当前目录下目录个数，不包含子目录
#ll |grep "^d"|wc -l
#当前目录下文件个数，包含子目录
#ll -R|grep "^-"|wc -l
#当前目录下目录个数，包含子目录
#ll -R|grep "^d"|wc -l
