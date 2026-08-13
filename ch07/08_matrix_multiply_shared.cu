// chapter7 片段 8（自 src/chapter7.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(err) << std::endl;                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }
// 核函数：计算 A^T * A 的块矩阵乘法
__global__ void matrixMultiplyShared(float *A, float *result, int N) {
    __shared__ float sharedA[32][32]; // 定义共享内存块
    __shared__ float sharedB[32][32]; // A的转置部分存储
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float value = 0.0;
    for (int block = 0; block < (N + 31) / 32; ++block) {
        if (row < N && block * 32 + tx < N) {
            sharedA[ty][tx] = A[row * N + block * 32 + tx];
        } else {
            sharedA[ty][tx] = 0.0;
        }
        if (col < N && block * 32 + ty < N) {
            sharedB[ty][tx] = A[(block * 32 + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0;
        }
        __syncthreads();
        for (int i = 0; i < 32; ++i) {
            value += sharedA[ty][i] * sharedB[i][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        result[row * N + col] = value;
    }
}
int main() {
    const int N = 64; // 矩阵维度
    const size_t bytes = N * N * sizeof(float);
    // 主机内存分配并初始化
    float *hostA = new float[N * N];
    float *hostResult = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        hostA[i] = static_cast<float>(rand() % 100) / 10.0; // 随机初始化
    }
    // 打印输入矩阵
    std::cout << "Input Matrix A:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostA[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    // 设备内存分配
    float *deviceA, *deviceResult;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceResult, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 blockSize(32, 32);
    dim3 gridSize((N + 31) / 32, (N + 31) / 32);
    // 执行核函数
    matrixMultiplyShared<<<gridSize, blockSize>>>(deviceA, deviceResult, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostResult, deviceResult, bytes, cudaMemcpyDeviceToHost));
    // 打印结果矩阵
    std::cout << "Result Matrix A^T * A:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostResult[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    // 清理内存
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceResult));
    delete[] hostA;
    delete[] hostResult;
    return 0;
}
