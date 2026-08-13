// chapter1 片段 6a：单独的核函数，用 nvcc -ptx 编译成 add.ptx 供驱动API加载
// 修正：原文用 cuModuleGetFunction(..., "addKernel") 查找，
// 但 C++ 编译出的 PTX 入口名会被 name mangling 改写，故加 extern "C"。
#include <cuda_runtime.h>
extern "C" __global__ void addKernel(int *c, const int *a, const int *b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
