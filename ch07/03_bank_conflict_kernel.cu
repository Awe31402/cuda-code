// chapter7 片段 3（自 src/chapter7.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(err) << std::endl;                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }
// 核函数：共享内存存在Bank冲突
__global__ void bankConflictKernel(float *output, int N) {
    __shared__ float sharedMemory[32]; // 每个Bank对应一个float
    int tid = threadIdx.x;
    // Bank冲突：所有线程访问同一Bank不同地址
    sharedMemory[tid * 2 % 32] = tid;
    __syncthreads();
    if (tid < N) {
        output[tid] = sharedMemory[tid * 2 % 32];
    }
}
// 核函数：优化后的共享内存访问
__global__ void optimizedKernel(float *output, int N) {
    __shared__ float sharedMemory[32]; // 每个Bank对应一个float
    int tid = threadIdx.x;
    // 无Bank冲突：访问模式调整为对齐方式
    sharedMemory[tid] = tid;
    __syncthreads();
    if (tid < N) {
        output[tid] = sharedMemory[tid];
    }
}
int main() {
    const int N = 32;
    const size_t bytes = N * sizeof(float);
    // 主机内存分配
    float *hostOutput = new float[N];
    // 设备内存分配
    float *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    // 设置线程块大小
    const int blockSize = 32;
    const int gridSize = 1;
    // 执行含Bank冲突的核函数
    bankConflictKernel<<<gridSize, blockSize>>>(deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    std::cout << "Output with Bank Conflict:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;
    // 执行优化后的核函数
    optimizedKernel<<<gridSize, blockSize>>>(deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    std::cout << "Output without Bank Conflict:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    CUDA_CHECK(cudaFree(deviceOutput));
    delete[] hostOutput;
    return 0;
}
