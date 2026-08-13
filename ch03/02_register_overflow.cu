// chapter3 片段 2（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数：未优化，导致寄存器溢出到局部内存
__global__ void registerOverflow(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int temp[64]; // 大量局部数组使用
        for (int i = 0; i < 64; ++i) {
            temp[i] = idx * i;
        }
        int sum = 0;
        for (int i = 0; i < 64; ++i) {
            sum += temp[i];
        }
        data[idx] = sum;
    }
}
// 核函数：优化版本，避免局部内存溢出
__global__ void optimizedRegisterUsage(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int sum = 0;
        for (int i = 0; i < 64; ++i) {
            sum += idx * i; // 消除局部数组，直接计算
        }
        data[idx] = sum;
    }
}
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << " 错误: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    const int N = 1 << 20; // 数据量 (1M元素)
    const int bytes = N * sizeof(int);
    int *hostData = new int[N];
    int *deviceData;
    // 分配设备内存
    cudaMalloc(&deviceData, bytes);
    checkCudaError("设备内存分配失败");
    // 配置线程块和网格
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    // 未优化版本计时
    auto start = std::chrono::high_resolution_clock::now();
    registerOverflow<<<gridDim, blockDim>>>(deviceData, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto overflowDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 优化版本计时
    start = std::chrono::high_resolution_clock::now();
    optimizedRegisterUsage<<<gridDim, blockDim>>>(deviceData, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto optimizedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 输出结果
    std::cout << "未优化版本执行时间: " << overflowDuration << " ms" << std::endl;
    std::cout << "优化版本执行时间: " << optimizedDuration << " ms" << std::endl;
    // 释放资源
    cudaFree(deviceData);
    delete[] hostData;
    return 0;
}
