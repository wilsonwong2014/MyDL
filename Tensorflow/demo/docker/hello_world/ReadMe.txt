【docker】docker建立最简单最小的helloworld镜像
    https://blog.csdn.net/u012819339/article/details/80007919

    查看hello链接了哪些库文件
    $ ldd hello

===============================================
1.编写c源文件hello.c
#include <stdio.h>

int main(int argc, char *argv[])
{
        printf("hello world\n");
        return 0;
}

2. gcc静态编译
  $gcc -static -Os -nostartfiles -fno-asynchronous-unwind-tables -o hello hello.c
  $gcc -static -o hello hello.c  
  $ls hello -al

3. 编写Dockerfile
FROM scratch
COPY hello /
CMD ["/hello"]


4.编译镜像
    $docker build -t myhello:1.0 .

5.测试镜像
    $docker run --rm myhello:1.0



