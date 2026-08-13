// chapter7 片段 2（自 src/chapter7.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(err) << std::endl;                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }
// 核函数：非对齐访问
__global__ void nonCoalescedAccess(const float *input, float *output, int N, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = input[tid * stride];
    }
}
// 核函数：对齐访问优化
__global__ void coalescedAccess(const float *input, float *output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = input[tid];
    }
}
int main() {
    const int N = 1 << 20; // 1M元素
    const int stride = 2; // 非对齐步长
    const size_t bytes = N * sizeof(float);
    // 主机内存分配
    std::vector<float> hostInput(N * stride, 1.0f);
    std::vector<float> hostOutput(N);
    // 设备内存分配
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, N * stride * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), N * stride * sizeof(float), cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    // 非对齐访问
    nonCoalescedAccess<<<gridSize, blockSize>>>(deviceInput, deviceOutput, N, stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));
    // 打印非对齐结果
    std::cout << "Output (non-coalesced): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;
    // 对齐访问
    coalescedAccess<<<gridSize, blockSize>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));
    // 打印对齐结果
    std::cout << "Output (coalesced): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    return 0;
}
