基于busybox构建最小linux Docker镜像系统
    https://blog.csdn.net/hknaruto/article/details/70229896

===========================================================
1. 创建linux系统的基础目录
    $mkdir usr 
    $mkdir usr/lib 
    $mkdir usr/local 
    $mkdir usr/lib64 
    $mkdir usr/sbin 
    $mkdir usr/include 
    $mkdir usr/bin 
    $mkdir var/ 
    $mkdir var/lib 
    $mkdir var/run 
    $mkdir var/local 
    $mkdir var/log 
    $mkdir tmp

2. 创建 /lib /lib64软链接 
    $ln -s usr/lib lib
    $ln -s usr/lib64 lib64
    $ln -s usr/bin bin

3. 下载busybox并拷贝至 usr/bin目录 
    $curl -O https://busybox.net/downloads/binaries/1.21.1/busybox-x86_64
    $mv busybox-x86_64 usr/bin/busybox

4. 生成系统指令软链接 
    $chmod +x usr/bin/busybox
    $./usr/bin/busybox 

    利用http://tool.oschina.net/regex/ 提供的在线正则，正则表达式： \s*([^,]+),?\s*， 替换文本：ln -s busybox $1;  最终得到指令如下 
    $cd usr/bin
    $ln -s  busybox [;ln -s  busybox [[;ln -s  busybox acpid;ln -s  busybox add-shell;ln -s  busybox addgroup;ln -s  busybox adduser;ln -s  busybox adjtimex;ln -s  busybox arp;ln -s  busybox arping;ln -s  busybox ash;ln -s  busybox awk;ln -s  busybox base64;ln -s  busybox basename;ln -s  busybox beep;ln -s  busybox blkid;ln -s  busybox blockdev;ln -s  busybox bootchartd;ln -s  busybox brctl;ln -s  busybox bunzip2;ln -s  busybox bzcat;ln -s  busybox bzip2;ln -s  busybox cal;ln -s  busybox cat;ln -s  busybox catv;ln -s  busybox chat;ln -s  busybox chattr;ln -s  busybox chgrp;ln -s  busybox chmod;ln -s  busybox chown;ln -s  busybox chpasswd;ln -s  busybox chpst;ln -s  busybox chroot;ln -s  busybox chrt;ln -s  busybox chvt;ln -s  busybox cksum;ln -s  busybox clear;ln -s  busybox cmp;ln -s  busybox comm;ln -s  busybox conspy;ln -s  busybox cp;ln -s  busybox cpio;ln -s  busybox crond;ln -s  busybox crontab;ln -s  busybox cryptpw;ln -s  busybox cttyhack;ln -s  busybox cut;ln -s  busybox date;ln -s  busybox dc;ln -s  busybox dd;ln -s  busybox deallocvt;ln -s  busybox delgroup;ln -s  busybox deluser;ln -s  busybox depmod;ln -s  busybox devmem;ln -s  busybox df;ln -s  busybox dhcprelay;ln -s  busybox diff;ln -s  busybox dirname;ln -s  busybox dmesg;ln -s  busybox dnsd;ln -s  busybox dnsdomainname;ln -s  busybox dos2unix;ln -s  busybox du;ln -s  busybox dumpkmap;ln -s  busybox dumpleases;ln -s  busybox echo;ln -s  busybox ed;ln -s  busybox egrep;ln -s  busybox eject;ln -s  busybox env;ln -s  busybox envdir;ln -s  busybox envuidgid;ln -s  busybox ether-wake;ln -s  busybox expand;ln -s  busybox expr;ln -s  busybox fakeidentd;ln -s  busybox false;ln -s  busybox fbset;ln -s  busybox fbsplash;ln -s  busybox fdflush;ln -s  busybox fdformat;ln -s  busybox fdisk;ln -s  busybox fgconsole;ln -s  busybox fgrep;ln -s  busybox find;ln -s  busybox findfs;ln -s  busybox flock;ln -s  busybox fold;ln -s  busybox free;ln -s  busybox freeramdisk;ln -s  busybox fsck;ln -s  busybox fsck.minix;ln -s  busybox fsync;ln -s  busybox ftpd;ln -s  busybox ftpget;ln -s  busybox ftpput;ln -s  busybox fuser;ln -s  busybox getopt;ln -s  busybox getty;ln -s  busybox grep;ln -s  busybox groups;ln -s  busybox gunzip;ln -s  busybox gzip;ln -s  busybox halt;ln -s  busybox hd;ln -s  busybox hdparm;ln -s  busybox head;ln -s  busybox hexdump;ln -s  busybox hostid;ln -s  busybox hostname;ln -s  busybox httpd;ln -s  busybox hush;ln -s  busybox hwclock;ln -s  busybox id;ln -s  busybox ifconfig;ln -s  busybox ifdown;ln -s  busybox ifenslave;ln -s  busybox ifplugd;ln -s  busybox ifup;ln -s  busybox inetd;ln -s  busybox init;ln -s  busybox insmod;ln -s  busybox install;ln -s  busybox ionice;ln -s  busybox iostat;ln -s  busybox ip;ln -s  busybox ipaddr;ln -s  busybox ipcalc;ln -s  busybox ipcrm;ln -s  busybox ipcs;ln -s  busybox iplink;ln -s  busybox iproute;ln -s  busybox iprule;ln -s  busybox iptunnel;ln -s  busybox kbd_mode;ln -s  busybox kill;ln -s  busybox killall;ln -s  busybox killall5;ln -s  busybox klogd;ln -s  busybox last;ln -s  busybox less;ln -s  busybox linux32;ln -s  busybox linux64;ln -s  busybox linuxrc;ln -s  busybox ln;ln -s  busybox loadfont;ln -s  busybox loadkmap;ln -s  busybox logger;ln -s  busybox login;ln -s  busybox logname;ln -s  busybox logread;ln -s  busybox losetup;ln -s  busybox lpd;ln -s  busybox lpq;ln -s  busybox lpr;ln -s  busybox ls;ln -s  busybox lsattr;ln -s  busybox lsmod;ln -s  busybox lsof;ln -s  busybox lspci;ln -s  busybox lsusb;ln -s  busybox lzcat;ln -s  busybox lzma;ln -s  busybox lzop;ln -s  busybox lzopcat;ln -s  busybox makedevs;ln -s  busybox makemime;ln -s  busybox man;ln -s  busybox md5sum;ln -s  busybox mdev;ln -s  busybox mesg;ln -s  busybox microcom;ln -s  busybox mkdir;ln -s  busybox mkdosfs;ln -s  busybox mke2fs;ln -s  busybox mkfifo;ln -s  busybox mkfs.ext2;ln -s  busybox mkfs.minix;ln -s  busybox mkfs.vfat;ln -s  busybox mknod;ln -s  busybox mkpasswd;ln -s  busybox mkswap;ln -s  busybox mktemp;ln -s  busybox modinfo;ln -s  busybox modprobe;ln -s  busybox more;ln -s  busybox mount;ln -s  busybox mountpoint;ln -s  busybox mpstat;ln -s  busybox mt;ln -s  busybox mv;ln -s  busybox nameif;ln -s  busybox nanddump;ln -s  busybox nandwrite;ln -s  busybox nbd-client;ln -s  busybox nc;ln -s  busybox netstat;ln -s  busybox nice;ln -s  busybox nmeter;ln -s  busybox nohup;ln -s  busybox nslookup;ln -s  busybox ntpd;ln -s  busybox od;ln -s  busybox openvt;ln -s  busybox passwd;ln -s  busybox patch;ln -s  busybox pgrep;ln -s  busybox pidof;ln -s  busybox ping;ln -s  busybox ping6;ln -s  busybox pipe_progress;ln -s  busybox pivot_root;ln -s  busybox pkill;ln -s  busybox pmap;ln -s  busybox popmaildir;ln -s  busybox poweroff;ln -s  busybox powertop;ln -s  busybox printenv;ln -s  busybox printf;ln -s  busybox ps;ln -s  busybox pscan;ln -s  busybox pstree;ln -s  busybox pwd;ln -s  busybox pwdx;ln -s  busybox raidautorun;ln -s  busybox rdate;ln -s  busybox rdev;ln -s  busybox readahead;ln -s  busybox readlink;ln -s  busybox readprofile;ln -s  busybox realpath;ln -s  busybox reboot;ln -s  busybox reformime;ln -s  busybox remove-shell;ln -s  busybox renice;ln -s  busybox reset;ln -s  busybox resize;ln -s  busybox rev;ln -s  busybox rm;ln -s  busybox rmdir;ln -s  busybox rmmod;ln -s  busybox route;ln -s  busybox rpm;ln -s  busybox rpm2cpio;ln -s  busybox rtcwake;ln -s  busybox run-parts;ln -s  busybox runlevel;ln -s  busybox runsv;ln -s  busybox runsvdir;ln -s  busybox rx;ln -s  busybox script;ln -s  busybox scriptreplay;ln -s  busybox sed;ln -s  busybox sendmail;ln -s  busybox seq;ln -s  busybox setarch;ln -s  busybox setconsole;ln -s  busybox setfont;ln -s  busybox setkeycodes;ln -s  busybox setlogcons;ln -s  busybox setserial;ln -s  busybox setsid;ln -s  busybox setuidgid;ln -s  busybox sh;ln -s  busybox sha1sum;ln -s  busybox sha256sum;ln -s  busybox sha3sum;ln -s  busybox sha512sum;ln -s  busybox showkey;ln -s  busybox slattach;ln -s  busybox sleep;ln -s  busybox smemcap;ln -s  busybox softlimit;ln -s  busybox sort;ln -s  busybox split;ln -s  busybox start-stop-daemon;ln -s  busybox stat;ln -s  busybox strings;ln -s  busybox stty;ln -s  busybox su;ln -s  busybox sulogin;ln -s  busybox sum;ln -s  busybox sv;ln -s  busybox svlogd;ln -s  busybox swapoff;ln -s  busybox swapon;ln -s  busybox switch_root;ln -s  busybox sync;ln -s  busybox sysctl;ln -s  busybox syslogd;ln -s  busybox tac;ln -s  busybox tail;ln -s  busybox tar;ln -s  busybox tcpsvd;ln -s  busybox tee;ln -s  busybox telnet;ln -s  busybox telnetd;ln -s  busybox test;ln -s  busybox tftp;ln -s  busybox tftpd;ln -s  busybox time;ln -s  busybox timeout;ln -s  busybox top;ln -s  busybox touch;ln -s  busybox tr;ln -s  busybox traceroute;ln -s  busybox traceroute6;ln -s  busybox true;ln -s  busybox tty;ln -s  busybox ttysize;ln -s  busybox tunctl;ln -s  busybox udhcpc;ln -s  busybox udhcpd;ln -s  busybox udpsvd;ln -s  busybox umount;ln -s  busybox uname;ln -s  busybox unexpand;ln -s  busybox uniq;ln -s  busybox unix2dos;ln -s  busybox unlzma;ln -s  busybox unlzop;ln -s  busybox unxz;ln -s  busybox unzip;ln -s  busybox uptime;ln -s  busybox users;ln -s  busybox usleep;ln -s  busybox uudecode;ln -s  busybox uuencode;ln -s  busybox vconfig;ln -s  busybox vi;ln -s  busybox vlock;ln -s  busybox volname;ln -s  busybox wall;ln -s  busybox watch;ln -s  busybox watchdog;ln -s  busybox wc;ln -s  busybox wget;ln -s  busybox which;ln -s  busybox who;ln -s  busybox whoami;ln -s  busybox whois;ln -s  busybox xargs;ln -s  busybox xz;ln -s  busybox xzcat;ln -s  busybox yes;ln -s  busybox zcat;ln -s  busybox zcip ;

    $cd -;

    执行指令，建立软链接 

5. 编辑Dockerfile
FROM scratch
MAINTAINER Wilson.Wong.
ADD ./ /
RUN rm /Dockerfile

6. 制作镜像 
    $docker build -t minos .

7. 查看镜像
    $docker images |  grep minos

8. 测试镜像 
    $docker run --rm -it minos /bin/sh



