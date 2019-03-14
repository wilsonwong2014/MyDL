#!/bin/bash
#递归遍历文件夹
travFolder()
{ 
    #echo "travFolder"
    flist=`ls $1`
    cd $1
    #echo $flist
    for f in $flist
    do
        if test -d $f
        then
            echo "dir:$f"
            travFolder $f
        else
            echo "file:$f"
            #changeName $f
        fi

    done
    cd ../ 
}

#测试
travFolder ~/MyPrjs/script
