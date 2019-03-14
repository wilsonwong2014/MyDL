#!/usr/bin/usr sh

# 检索软件安装信息 并输出文件
# 使用范例：
# $bash list_soft.sh filename "command line"
#
#

sdate=`date +%Y_%m_%d_%H_%M_%S`
slogfile="$HOME/work/hjw/log/log_`date +%Y_%m_%d_%H_%M_%S`_$1"
scmd="$2"
echo "$slogfile"
echo "$scmd"
echo $sdate
echo "==============================================================" >> $slogfile
echo "$sdate" >> $slogfile
echo "command:$scmd" >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "dpkg -l" >> $slogfile
dpkg -l >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "pip list" >> $slogfile
pip list >>$slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "pip3 list" >> $slogfile
pip3 list >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "dpkg -L python" >> $slogfile
dpkg -L python >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "dpkg -L python3" >> $slogfile
dpkg -L python3 >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "dpkg -L python3.5" >> $slogfile
dpkg -L python3.5 >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "dpkg -L python-pip" >> $slogfile
dpkg -L python-pip >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile
echo "dpkg -L python3-pip" >> $slogfile
dpkg -L python3-pip >> $slogfile
echo "--------------------------------------------------------------" >> $slogfile



