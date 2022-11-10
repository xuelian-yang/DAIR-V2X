
import logging
import sys
import os
import os.path as osp

if __name__ == "__main__":
    path = "./"
    for root, dirs, files in os.walk(path):
        for i in files:
            if i.endswith(".json"):
                print(i)


'''

conda config --show channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
 
conda install pytorch torchvision cudatoolkit=10.0  # 删除安装命令最后的 -c pytorch，才会采用清华源安装。
'''