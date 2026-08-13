// chapter3 片段 8（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define TILE_SIZE 32
// 非优化版本：全局内存直接访问
__global__ void matrixAddGlobal(const float *a, const float *b, float *c, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        c[y * N + x] = a[y * N + x] + b[y * N + x]; // 非合并访问
    }
}
// 优化版本：共享内存优化
__global__ void matrixAddShared(const float *a, const float *b, float *c, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (x < N && y < N) {
        tileA[ty][tx] = a[y * N + x];
        tileB[ty][tx] = b[y * N + x];
    }
    __syncthreads();
    if (x < N && y < N) {
        c[y * N + x] = tileA[ty][tx] + tileB[ty][tx];
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
    float *hostCGlobal = new float[size];
    float *hostCShared = new float[size];
    for (int i = 0; i < size; ++i) {
        hostA[i] = static_cast<float>(i % 100);
        hostB[i] = static_cast<float>(i % 200);
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
    matrixAddGlobal<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto globalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostCGlobal, deviceC, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    start = std::chrono::high_resolution_clock::now();
    matrixAddShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto sharedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostCShared, deviceC, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (hostCGlobal[i] != hostCShared[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "非优化版本执行时间: " << globalDuration << " ms" << std::endl;
    std::cout << "共享内存优化版本执行时间: " << sharedDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostCGlobal;
    delete[] hostCShared;
    return 0;
}
