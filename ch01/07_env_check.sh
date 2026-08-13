#!/usr/bin/env bash
# chapter1 片段 7：驱动/工具链版本匹配检查
set -u

nvidia-smi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 若架构与显卡不符会编译或运行失败
# nvcc -arch=sm_75 -code=sm_75,compute_75 program.cu

nvcc --version

# 原文示例输出：
# NVIDIA-SMI 470.57.02
# Driver Version: 470.57.02
# CUDA Version: 11.4

# nvcc -arch=sm_70 program.cu -o program
# ./program
# export PATH=/usr/local/cuda-11.4/bin:$PATH
