// chapter6 片段 7（自 src/chapter6.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
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
// 核函数：矩阵乘法
__global__ void matrixMultiply(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
int main() {
    const int N = 1024; // 矩阵大小
    const size_t bytes = N * N * sizeof(float);
    // 主机内存分配
    std::vector<float> hostA(N * N, 1.0f);
    std::vector<float> hostB(N * N, 1.0f);
    std::vector<float> hostC(N * N, 0.0f);
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 threads(16, 16); // 线程块大小
    dim3 grid((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    // 启动核函数
    matrixMultiply<<<grid, threads>>>(deviceA, deviceB, deviceC, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
    // 打印部分结果
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << hostC[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    return 0;
}
