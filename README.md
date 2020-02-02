## 先来配置pytorch环境

1. 如果用conda下载安装包较慢，可以用清华源替代默认的conda源

   `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ `

   `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`

   `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ `

2. 创建conda虚拟环境，并激活

   `conda create -n cifar python=3.6`

   `activate cifar`

3. 根据官网提示安装pytorch，这里安装cpu版本

   `conda install pytorch torchvision cpuonly`

   使用7.0版本的pillow可能出现Import Error，可以通过装低版本解决

   `conda install pillow==6`

   安装matplotlib库

   `conda install matplotlib`

## Pytorch代码

跟着官网例子一步步走下来，cifar10数据集官方下载较慢，可以手动下载

运行代码查看效果

`` python main.py``

官网例子跑下来是53%准确率，稍微修改了网络参数后有68%

后面有空试试其他参数，cifar10的top-1准确率90%应该很轻松的

## ResNet

网上找了个resnet18的代码，我的小笔电直接爆炸了