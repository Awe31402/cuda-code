// chapter5 片段 4（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 核函数：数据竞争演示
__global__ void incrementArrayRaceCondition(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[0] += 1; // 产生数据竞争
    }
}
// 核函数：使用原子操作解决数据竞争
__global__ void incrementArrayAtomic(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(&data[0], 1); // 使用原子操作消除数据竞争
    }
}
int main() {
    const int N = 1024; // 总线程数
    const int bytes = sizeof(int);
    // 分配设备内存
    int *deviceData;
    cudaMalloc(&deviceData, bytes);
    // 初始化数据
    int initialValue = 0;
    cudaMemcpy(deviceData, &initialValue, bytes, cudaMemcpyHostToDevice);
    // 设置线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    std::cout << "运行产生数据竞争的核函数..." << std::endl;
    incrementArrayRaceCondition<<<gridSize, blockSize>>>(deviceData, N);
    cudaDeviceSynchronize();
    // 检查结果
    int result;
    cudaMemcpy(&result, deviceData, bytes, cudaMemcpyDeviceToHost);
    std::cout << "数据竞争情况下的结果: " << result << " (期望值: " << N << ")" << std::endl;
    // 重置数据
    cudaMemcpy(deviceData, &initialValue, bytes, cudaMemcpyHostToDevice);
    std::cout << "运行使用原子操作的核函数..." << std::endl;
    incrementArrayAtomic<<<gridSize, blockSize>>>(deviceData, N);
    cudaDeviceSynchronize();
    // 检查结果
    cudaMemcpy(&result, deviceData, bytes, cudaMemcpyDeviceToHost);
    std::cout << "使用原子操作情况下的结果: " << result << " (期望值: " << N << ")" << std::endl;
    // 释放设备内存
    cudaFree(deviceData);
    return 0;
}
