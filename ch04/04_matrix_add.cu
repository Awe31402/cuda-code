// chapter4 片段 4（自 src/chapter4.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define N 1024 // 矩阵大小
// 核函数：简单的矩阵加法
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
    // 分配主机内存
    float *hostA, *hostB, *hostC;
    cudaHostAlloc(&hostA, bytes, cudaHostAllocDefault); // 锁页内存
    cudaHostAlloc(&hostB, bytes, cudaHostAllocDefault); // 锁页内存
    cudaHostAlloc(&hostC, bytes, cudaHostAllocDefault); // 锁页内存
    // 初始化主机内存
    for (int i = 0; i < size; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    checkCudaError("设备内存分配失败");
    // 创建CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    // 异步数据传输与核函数执行
    auto start = std::chrono::high_resolution_clock::now();
    // 主机到设备的异步数据传输
    cudaMemcpyAsync(deviceA, hostA, bytes, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceB, hostB, bytes, cudaMemcpyHostToDevice, stream2);
    // 等待传输完成后启动核函数
    matrixAdd<<<gridDim, blockDim, 0, stream1>>>(deviceA, deviceB, deviceC, N);
    checkCudaError("核函数执行失败");
    // 设备到主机的异步数据传输
    cudaMemcpyAsync(hostC, deviceC, bytes, cudaMemcpyDeviceToHost, stream1);
    // 同步流，确保所有操作完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 验证结果
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (hostC[i] != hostA[i] + hostB[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "异步数据传输与核函数重叠执行时间: " << duration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    // 清理资源
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return 0;
}
