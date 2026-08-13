// chapter4 片段 2（自 src/chapter4.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define N 1024 // 矩阵大小
// 核函数：非优化版本
__global__ void matrixAddBasic(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}
// 核函数：优化版本，利用共享内存和线程调度
__global__ void matrixAddOptimized(const float *a, const float *b, float *c, int n) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (row < n && col < n) {
        tileA[ty][tx] = a[row * n + col];
        tileB[ty][tx] = b[row * n + col];
    } else {
        tileA[ty][tx] = 0.0f;
        tileB[ty][tx] = 0.0f;
    }
    __syncthreads();
    if (row < n && col < n) {
        c[row * n + col] = tileA[ty][tx] + tileB[ty][tx];
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
    float *hostA = new float[size];
    float *hostB = new float[size];
    float *hostCBasic = new float[size];
    float *hostCOptimized = new float[size];
    for (int i = 0; i < size; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    checkCudaError("设备内存分配失败");
    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddBasic<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto basicDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostCBasic, deviceC, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    start = std::chrono::high_resolution_clock::now();
    matrixAddOptimized<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto optimizedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostCOptimized, deviceC, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (hostCBasic[i] != hostCOptimized[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "非优化版本执行时间: " << basicDuration << " ms" << std::endl;
    std::cout << "优化版本执行时间: " << optimizedDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostCBasic;
    delete[] hostCOptimized;
    return 0;
}
