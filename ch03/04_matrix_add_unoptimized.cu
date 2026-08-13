// chapter3 片段 4（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define TILE_SIZE 32
// 未优化的全局内存访问
__global__ void matrixAddUnoptimized(const float *a, const float *b, float *c, int N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N && y < N) {
        c[y * N + x] = a[y * N + x] + b[y * N + x]; // 非合并访问
    }
}
// 优化的全局内存访问
__global__ void matrixAddOptimized(const float *a, const float *b, float *c, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (x < N && y < N) {
        tileA[ty][tx] = a[y * N + x]; // 合并访问加载
        tileB[ty][tx] = b[y * N + x];
    }
    __syncthreads();
    if (x < N && y < N) {
        c[y * N + x] = tileA[ty][tx] + tileB[ty][tx]; // 合并访问存储
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
    const int N = 1024;
    const int size = N * N;
    const int bytes = size * sizeof(float);
    float *hostA = new float[size];
    float *hostB = new float[size];
    float *hostCUnoptimized = new float[size];
    float *hostCOptimized = new float[size];
    for (int i = 0; i < size; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(size - i);
    }
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    checkCudaError("设备内存分配失败");
    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddUnoptimized<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto unoptimizedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostCUnoptimized, deviceC, bytes, cudaMemcpyDeviceToHost);
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
        if (hostCUnoptimized[i] != hostCOptimized[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "未优化版本执行时间: " << unoptimizedDuration << " ms" << std::endl;
    std::cout << "优化版本执行时间: " << optimizedDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostCUnoptimized;
    delete[] hostCOptimized;
    return 0;
}
