#!/usr/bin/env bash
# chapter1 片段 4：nvcc 常用编译选项（参考用，需要 program.cu 才能执行）
set -u

# 指定目标架构
nvcc -arch=sm_75 program.cu -o program
# 生成 PTX 中间代码
nvcc program.cu -ptx -o program.ptx
# 限制每线程寄存器数量
nvcc program.cu -maxrregcount=32 -o program
# 生成设备端调试信息
nvcc program.cu -G -o program
# 同时嵌入 SASS 与 PTX
nvcc program.cu -arch=compute_75 -code=sm_75,compute_75 -o program
# 打印寄存器/共享内存使用量
nvcc program.cu -Xptxas=-v -o program
