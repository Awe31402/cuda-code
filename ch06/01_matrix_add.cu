// chapter6 片段 1（自 src/chapter6.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 宏函数：通用错误检测
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(err) << std::endl;                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }
// 核函数：矩阵加法
__global__ void matrixAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    const int N = 1 << 20; // 矩阵大小
    const int bytes = N * sizeof(float);
    // 主机内存分配
    float *hostA = new float[N];
    float *hostB = new float[N];
    float *hostC = new float[N];
    for (int i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CUDA_CHECK(cudaMalloc((void **)&deviceA, bytes));
    CUDA_CHECK(cudaMalloc((void **)&deviceB, bytes));
    CUDA_CHECK(cudaMalloc((void **)&deviceC, bytes));
    // 测试数据传输时间
    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    double transferTime = std::chrono::duration<double, std::milli>(end - start).count();
    // 设置线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    // 测试计算时间
    start = std::chrono::high_resolution_clock::now();
    matrixAdd<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double computeTime = std::chrono::duration<double, std::milli>(end - start).count();
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost));
    // 输出结果
    std::cout << "Transfer Time (ms): " << transferTime << std::endl;
    std::cout << "Compute Time (ms): " << computeTime << std::endl;
    std::cout << "Data Transfer to Compute Ratio: " << transferTime / computeTime << std::endl;
    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "]: " << hostC[i] << std::endl;
    }
    // 释放内存
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
