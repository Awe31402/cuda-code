// chapter4 片段 7（自 src/chapter4.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define N 1024 // 矩阵大小
// 核函数：矩阵乘法
__global__ void matrixMultiply(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
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
    // 分配锁页内存
    float *hostA, *hostB, *hostC;
    cudaHostAlloc(&hostA, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hostB, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hostC, bytes, cudaHostAllocDefault);
    // 初始化数据
    for (int i = 0; i < size; ++i) {
        hostA[i] = static_cast<float>(i % 100);
        hostB[i] = static_cast<float>((i % 100) * 2);
    }
    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    // 创建CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    // 异步传输锁页内存到设备
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(deviceA, hostA, bytes, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceB, hostB, bytes, cudaMemcpyHostToDevice, stream2);
    // 核函数执行
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixMultiply<<<gridDim, blockDim, 0, stream1>>>(deviceA, deviceB, deviceC, N);
    // 异步传输设备到主机
    cudaMemcpyAsync(hostC, deviceC, bytes, cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    auto end = std::chrono::high_resolution_clock::now();
    auto lockedMemoryDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 清理锁页内存
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
    // 使用Unified Memory
    float *unifiedA, *unifiedB, *unifiedC;
    cudaMallocManaged(&unifiedA, bytes);
    cudaMallocManaged(&unifiedB, bytes);
    cudaMallocManaged(&unifiedC, bytes);
    // 初始化Unified Memory
    for (int i = 0; i < size; ++i) {
        unifiedA[i] = static_cast<float>(i % 100);
        unifiedB[i] = static_cast<float>((i % 100) * 2);
    }
    // 执行核函数
    start = std::chrono::high_resolution_clock::now();
    matrixMultiply<<<gridDim, blockDim>>>(unifiedA, unifiedB, unifiedC, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto unifiedMemoryDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 验证结果
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        float expected = 0.0f;
        for (int k = 0; k < N; ++k) {
            expected += unifiedA[(i / N) * N + k] * unifiedB[k * N + (i % N)];
        }
        if (abs(unifiedC[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "锁页内存执行时间: " << lockedMemoryDuration << " ms" << std::endl;
    std::cout << "Unified Memory执行时间: " << unifiedMemoryDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    // 清理Unified Memory资源
    cudaFree(unifiedA);
    cudaFree(unifiedB);
    cudaFree(unifiedC);
    return 0;
}
