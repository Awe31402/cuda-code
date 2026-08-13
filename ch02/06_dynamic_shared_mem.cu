// chapter2 片段 6：动态共享内存 + 寄存器使用
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数：简单计算模拟寄存器和共享内存使用
__global__ void computeKernel(int *output, int N) {
    extern __shared__ int sharedMemory[]; // 动态分配的共享内存
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // 全局索引
    if (idx < N) {
        // 使用共享内存
        sharedMemory[threadIdx.x] = idx % 10;
        // 模拟寄存器计算
        int localValue = sharedMemory[threadIdx.x];
        for (int i = 0; i < 1000; ++i) {
            localValue += i * threadIdx.x;
        }
        // 写入结果
        output[idx] = localValue;
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
    const int dataSize = 1 << 20; // 数据量 (1M元素)
    int *hostOutput = new int[dataSize]; // 主机结果数据
    int *deviceOutput;
    // 初始化主机数据
    for (int i = 0; i < dataSize; ++i) {
        hostOutput[i] = 0;
    }
    // 分配设备内存
    cudaMalloc(&deviceOutput, dataSize * sizeof(int));
    checkCudaError("设备内存分配失败");
    // 配置线程块大小
    int blockSizes[] = {32, 64, 128, 256, 512};
    const int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);
    // 测试不同线程块大小的性能
    for (int i = 0; i < numConfigs; ++i) {
        int blockSize = blockSizes[i];
        int numBlocks = (dataSize + blockSize - 1) / blockSize;
        std::cout << "线程块大小: " << blockSize << ", 网格大小: " << numBlocks << std::endl;
        // 动态分配共享内存大小
        int sharedMemorySize = blockSize * sizeof(int);
        auto start = std::chrono::high_resolution_clock::now();
        // 启动核函数
        computeKernel<<<numBlocks, blockSize, sharedMemorySize>>>(deviceOutput, dataSize);
        cudaDeviceSynchronize();
        checkCudaError("核函数执行失败");
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // 输出性能结果
        std::cout << "共享内存大小: " << sharedMemorySize << " 字节, 执行时间: " << duration << " ms" << std::endl;
    }
    // 将结果从设备拷贝回主机
    cudaMemcpy(hostOutput, deviceOutput, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 验证结果
    std::cout << "部分结果验证:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "hostOutput[" << i << "] = " << hostOutput[i] << std::endl;
    }
    // 释放内存
    cudaFree(deviceOutput);
    delete[] hostOutput;
    return 0;
}
