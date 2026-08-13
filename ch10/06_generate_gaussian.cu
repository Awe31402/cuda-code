// chapter10 片段 6（自 src/chapter10.txt 抽出）
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
// 核函数：生成高斯分布随机数
__global__ void generateGaussian(float* output, int size, unsigned long seed, float mean, float stddev) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    // 初始化cuRAND状态
    curandState state;
    curand_init(seed, idx, 0, &state);
    // 生成高斯分布随机数
    output[idx] = mean + stddev * curand_normal(&state);
}
int main() {
    const int arraySize = 1024;
    const int blockSize = 256;
    const int numBlocks = (arraySize + blockSize - 1) / blockSize;
    // 分配主机和设备内存
    float* hostArray = new float[arraySize];
    float* deviceArray;
    cudaMalloc(&deviceArray, arraySize * sizeof(float));
    // 调用核函数生成高斯分布随机数
    generateGaussian<<<numBlocks, blockSize>>>(deviceArray, arraySize, time(nullptr), 0.0f, 1.0f);
    // 将结果拷回主机
    cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
    // 输出部分随机数
    std::cout << "Generated Gaussian Random Numbers:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << hostArray[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    delete[] hostArray;
    cudaFree(deviceArray);
    return 0;
}
