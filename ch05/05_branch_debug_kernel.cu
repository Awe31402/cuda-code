// chapter5 片段 5（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 核函数：分支逻辑调试
__global__ void branchDebugKernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (data[idx] % 2 == 0) {
            printf("线程 %d 属于偶数分支，数据值: %d\n", idx, data[idx]);
        } else {
            printf("线程 %d 属于奇数分支，数据值: %d\n", idx, data[idx]);
        }
    }
}
int main() {
    const int N = 16;
    const int bytes = N * sizeof(int);
    // 分配主机内存并初始化数据
    int hostData[N];
    for (int i = 0; i < N; ++i) {
        hostData[i] = i + 1;
    }
    // 分配设备内存并拷贝数据
    int *deviceData;
    cudaMalloc(&deviceData, bytes);
    cudaMemcpy(deviceData, hostData, bytes, cudaMemcpyHostToDevice);
    // 启动核函数
    const int blockSize = 8;
    const int gridSize = (N + blockSize - 1) / blockSize;
    branchDebugKernel<<<gridSize, blockSize>>>(deviceData, N);
    cudaDeviceSynchronize();
    // 释放设备内存
    cudaFree(deviceData);
    return 0;
}
