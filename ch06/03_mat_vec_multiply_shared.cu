// chapter6 片段 3（自 src/chapter6.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
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
// 核函数：矩阵向量乘法
__global__ void matVecMultiplyShared(const float *matrix, const float *vector, float *result, int N) {
    __shared__ float sharedVector[1024]; // 共享内存，用于存储向量数据
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        // 加载向量到共享内存
        if (threadIdx.x < N) {
            sharedVector[threadIdx.x] = vector[threadIdx.x];
        }
        __syncthreads();
        // 计算矩阵行与向量的点积
        for (int i = 0; i < N; ++i) {
            sum += matrix[row * N + i] * sharedVector[i];
        }
        result[row] = sum;
    }
}
int main() {
    const int N = 1024; // 矩阵和向量大小
    const size_t matrixBytes = N * N * sizeof(float);
    const size_t vectorBytes = N * sizeof(float);
    const size_t resultBytes = N * sizeof(float);
    // 主机内存分配
    std::vector<float> hostMatrix(N * N, 1.0f);
    std::vector<float> hostVector(N, 1.0f);
    std::vector<float> hostResult(N, 0.0f);
    // 设备内存分配
    float *deviceMatrix, *deviceVector, *deviceResult;
    CUDA_CHECK(cudaMalloc(&deviceMatrix, matrixBytes));
    CUDA_CHECK(cudaMalloc(&deviceVector, vectorBytes));
    CUDA_CHECK(cudaMalloc(&deviceResult, resultBytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceMatrix, hostMatrix.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVector, hostVector.data(), vectorBytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    // 启动核函数
    matVecMultiplyShared<<<gridSize, blockSize>>>(deviceMatrix, deviceVector, deviceResult, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostResult.data(), deviceResult, resultBytes, cudaMemcpyDeviceToHost));
    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << "Result[" << i << "] = " << hostResult[i] << std::endl;
    }
    // 释放内存
    CUDA_CHECK(cudaFree(deviceMatrix));
    CUDA_CHECK(cudaFree(deviceVector));
    CUDA_CHECK(cudaFree(deviceResult));
    return 0;
}
