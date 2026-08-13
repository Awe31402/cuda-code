#!/usr/bin/env bash
# chapter1 片段 8/9：驱动切换、性能调优与 Docker/NVIDIA Container Toolkit 环境
set -u

# 切换 gcc/g++ 版本
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1

# 持久化模式与功耗限制
sudo nvidia-smi -pm 1
# sudo nvidia-smi -pl <功耗限制值>

# 性能分析
# nsys profile ./program

# 锁定时钟频率
# sudo nvidia-smi --applications-clocks=<显存频率>,<核心频率>

export CUDA_VISIBLE_DEVICES=0,1

# Windows/CMake 交叉参考
# cmake -G "Visual Studio 16 2019" -DCMAKE_CUDA_COMPILER=nvcc
# nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

# 重装驱动
# sudo apt-get --purge remove nvidia-*
# sudo apt-get install nvidia-driver-<版本号>

# 安装 Docker
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:11.4.2-base-ubuntu20.04 nvidia-smi

# 交互式开发容器
docker run --rm -it --gpus all nvidia/cuda:11.4.2-devel-ubuntu20.04 /bin/bash
docker build -t my-cuda-env .   # 需配合同目录 Dockerfile
docker run --rm -it --gpus all my-cuda-env /bin/bash
