// chapter6 片段 6（自 src/chapter6.txt 抽出）
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
// 核函数：指令融合与条件分支规约优化
__global__ void optimizedConditionFusion(const float *input, float *output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // 条件分支规约，通过条件运算符替代if-else语句
        output[tid] = (input[tid] > 0.5f) ? input[tid] * 2.0f : input[tid] * 0.5f;
    }
}
int main() {
    const int N = 1024; // 数据大小
    const size_t bytes = N * sizeof(float);
    // 主机内存分配
    std::vector<float> hostInput(N);
    std::vector<float> hostOutput(N);
    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        hostInput[i] = static_cast<float>(i) / N;
    }
    // 设备内存分配
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), bytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    // 启动核函数
    optimizedConditionFusion<<<gridSize, blockSize>>>(deviceInput, deviceOutput, N);
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
