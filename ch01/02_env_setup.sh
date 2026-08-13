#!/usr/bin/env bash
# chapter1 片段 2：CUDA 环境安装与检查命令（原文命令原样保留，仅作参考，不建议直接执行）
set -u

nvcc --version

# Windows 下的工具链路径（Linux 无效，仅记录）
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\libnvvp

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# source ~/.bashrc

nvidia-smi

# sudo apt install gcc g++
gcc --version

# CUDA samples 中的 deviceQuery
# make
# ./deviceQuery

# 编译并运行
# nvcc -o main main.cu
# ./main
