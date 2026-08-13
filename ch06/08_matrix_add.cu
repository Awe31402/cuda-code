// chapter6 片段 8（自 src/chapter6.txt 抽出）
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
// 核函数：矩阵加法
__global__ void matrixAdd(const float *A, const float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
int main() {
    int rows = 1024; // 矩阵行数
    int cols = 1024; // 矩阵列数
    const size_t bytes = rows * cols * sizeof(float);
    // 主机内存分配
    std::vector<float> hostA(rows * cols, 1.0f);
    std::vector<float> hostB(rows * cols, 2.0f);
    std::vector<float> hostC(rows * cols);
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));
    // 动态调整线程块和网格大小
    int blockSize = 16;
    if (rows * cols > 1 << 20) {
        blockSize = 32;
    }
    dim3 threads(blockSize, blockSize);
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    // 启动核函数
    matrixAdd<<<grid, threads>>>(deviceA, deviceB, deviceC, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
    // 打印部分结果
    std::cout << "Output: ";
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
