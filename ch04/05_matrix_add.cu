// chapter4 片段 5（自 src/chapter4.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define N 1024 // 矩阵大小
// 核函数：矩阵加法
__global__ void matrixAdd(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
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
    const int size = N * N;
    const int bytes = size * sizeof(float);
    // 分配传统显式内存
    float *hostA = new float[size];
    float *hostB = new float[size];
    float *hostC = new float[size];
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    checkCudaError("设备内存分配失败");
    // 初始化数据
    for (int i = 0; i < size; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    // 传统显式内存：主机到设备数据传输
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    // 执行核函数
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixAdd<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    checkCudaError("核函数执行失败");
    // 传统显式内存：设备到主机数据传输
    cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto traditionalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 清理传统显式内存资源
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    // 分配Unified Memory
    float *unifiedA, *unifiedB, *unifiedC;
    cudaMallocManaged(&unifiedA, bytes);
    cudaMallocManaged(&unifiedB, bytes);
    cudaMallocManaged(&unifiedC, bytes);
    checkCudaError("Unified Memory分配失败");
    // 初始化数据
    for (int i = 0; i < size; ++i) {
        unifiedA[i] = static_cast<float>(i);
        unifiedB[i] = static_cast<float>(i * 2);
    }
    // Unified Memory：核函数执行
    start = std::chrono::high_resolution_clock::now();
    matrixAdd<<<gridDim, blockDim>>>(unifiedA, unifiedB, unifiedC, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto unifiedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 清理Unified Memory资源
    cudaFree(unifiedA);
    cudaFree(unifiedB);
    cudaFree(unifiedC);
    // 输出性能结果
    std::cout << "传统显式内存执行时间: " << traditionalDuration << " ms" << std::endl;
    std::cout << "Unified Memory执行时间: " << unifiedDuration << " ms" << std::endl;
    return 0;
}
