// chapter6 片段 5（自 src/chapter6.txt 抽出）
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
// 核函数：优化后的Warp收敛
__global__ void optimizedWarpConvergence(const int *input, int *output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = tid / 32; // 计算Warp的ID
    if (tid < N) {
        int result = 0;
        // 避免分支发散，使用条件运算符
        result = (input[tid] % 2 == 0) ? input[tid] * 2 : input[tid] * 3;
        // 将结果写入输出数组
        output[tid] = result;
    }
}
int main() {
    const int N = 1024; // 数据大小
    const size_t bytes = N * sizeof(int);
    // 主机内存分配
    std::vector<int> hostInput(N);
    std::vector<int> hostOutput(N);
    for (int i = 0; i < N; ++i) {
        hostInput[i] = i;
    }
    // 设备内存分配
    int *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), bytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    // 启动核函数
    optimizedWarpConvergence<<<gridSize, blockSize>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));
    // 打印部分结果
    std::cout << "Output: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;
    // 释放内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    return 0;
}
