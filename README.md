## 先来配置pytorch环境

1. 如果用conda下载安装包较慢，可以用清华源替代默认的conda源

   ``conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ ``

2. 创建conda虚拟环境，并激活

   ``conda create -n cifar python=3.6
   activate cifar``

3. 根据官网提示安装pytorch，这里安装cpu版本

   使用7.0版本的pillow可能出现Import Error，可以通过装低版本解决

   ``conda install pytorch torchvision cpuonly
   conda install pillow==6``

## Pytorch代码

跟着官网例子一步步走下来，cifar数据集下载较慢不妨手动下载

运行代码查看效果

`` python main.py``

