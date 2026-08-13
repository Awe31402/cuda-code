// chapter3 片段 7（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define TILE_SIZE 32
// 核函数：默认缓存配置
__global__ void matrixMulDefault(const float *a, const float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}
// 核函数：共享内存优化
__global__ void matrixMulShared(const float *a, const float *b, float *c, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + tx < N) {
            tileA[ty][tx] = a[row * N + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        if (col < N && t * TILE_SIZE + ty < N) {
            tileB[ty][tx] = b[(t * TILE_SIZE + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        c[row * N + col] = sum;
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
    float *hostCDefault = new float[size];
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
    matrixMulDefault<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto defaultDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostCDefault, deviceC, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    start = std::chrono::high_resolution_clock::now();
    matrixMulShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto sharedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostCShared, deviceC, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (hostCDefault[i] != hostCShared[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "默认缓存配置版本执行时间: " << defaultDuration << " ms" << std::endl;
    std::cout << "共享内存优化版本执行时间: " << sharedDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostCDefault;
    delete[] hostCShared;
    return 0;
}
