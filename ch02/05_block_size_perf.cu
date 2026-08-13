// chapter2 片段 5：不同线程块大小的性能对比
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数：简单加法操作，模拟工作负载
__global__ void computeKernel(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // 计算线程的全局索引
    if (idx < N) {
        for (int i = 0; i < 1000; ++i) { // 模拟大量计算
            data[idx] += i;
        }
    }
}
// 检查CUDA错误
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << " 错误: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    // 配置参数
    const int dataSize = 1 << 20; // 数据量 (1M元素)
    int *hostData = new int[dataSize]; // 主机数据
    int *deviceData;
    // 初始化主机数据
    for (int i = 0; i < dataSize; ++i) {
        hostData[i] = 0;
    }
    // 分配设备内存
    cudaMalloc(&deviceData, dataSize * sizeof(int));
    checkCudaError("设备内存分配失败");
    // 将数据从主机拷贝到设备
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    // 配置不同线程块大小
    int blockSizes[] = {32, 64, 128, 256, 512};
    const int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);
    // 性能测试
    for (int i = 0; i < numConfigs; ++i) {
        int blockSize = blockSizes[i];
        int numBlocks = (dataSize + blockSize - 1) / blockSize; // 计算网格大小
        std::cout << "线程块大小: " << blockSize << ", 网格大小: " << numBlocks << std::endl;
        // 记录时间
        auto start = std::chrono::high_resolution_clock::now();
        // 启动核函数
        computeKernel<<<numBlocks, blockSize>>>(deviceData, dataSize);
        cudaDeviceSynchronize();
        checkCudaError("核函数执行失败");
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // 输出性能结果
        std::cout << "执行时间: " << duration << " ms" << std::endl;
    }
    // 将结果从设备拷贝回主机
    cudaMemcpy(hostData, deviceData, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 检查部分结果
    std::cout << "结果验证:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "hostData[" << i << "] = " << hostData[i] << std::endl;
    }
    // 释放内存
    cudaFree(deviceData);
    delete[] hostData;
    return 0;
}
