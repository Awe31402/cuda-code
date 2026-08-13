// chapter5 片段 6（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 核函数：演示分支发散
__global__ void branchDivergenceKernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (idx % 2 == 0) { // 偶数路径
            data[idx] = idx * 2;
        } else {            // 奇数路径
            data[idx] = idx * 3;
        }
    }
}
int main() {
    const int N = 1024;
    const int bytes = N * sizeof(int);
    int *hostData = new int[N];  // 分配主机内存
    int *deviceData;
    cudaMalloc(&deviceData, bytes);  // 分配设备内存
    const int blockSize = 256;  // 设置线程块大小
    const int gridSize = (N + blockSize - 1) / blockSize;
    branchDivergenceKernel<<<gridSize, blockSize>>>(deviceData, N);  // 启动核函数
    cudaDeviceSynchronize();
    cudaMemcpy(hostData, deviceData, bytes, cudaMemcpyDeviceToHost);  // 将结果从设备拷贝回主机
    for (int i = 0; i < 10; ++i) {  // 打印部分结果
        std::cout << "数据[" << i << "]: " << hostData[i] << std::endl;
    }
    cudaFree(deviceData);  // 释放设备内存
    delete[] hostData;  // 释放主机内存
    return 0;
}
