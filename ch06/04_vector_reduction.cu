// chapter6 片段 4（自 src/chapter6.txt 抽出）
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
// 核函数：使用共享内存实现向量归约
__global__ void vectorReduction(const float *input, float *output, int N) {
    extern __shared__ float sharedData[]; // 动态分配共享内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 将全局内存的数据加载到共享内存
    sharedData[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();
    // 在共享内存中进行归约操作
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    // 将每个块的归约结果写回全局内存
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
int main() {
    const int N = 1 << 20; // 向量大小
    const int blockSize = 256; // 每个块的线程数
    const int gridSize = (N + blockSize - 1) / blockSize; // 网格大小
    const size_t bytesInput = N * sizeof(float);
    const size_t bytesOutput = gridSize * sizeof(float);
    // 主机内存分配
    std::vector<float> hostInput(N, 1.0f); // 初始化向量，所有值为1
    std::vector<float> hostOutput(gridSize, 0.0f);
    // 设备内存分配
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytesInput));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytesOutput));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput.data(), bytesInput, cudaMemcpyHostToDevice));
    // 启动核函数
    size_t sharedMemoryBytes = blockSize * sizeof(float); // 每个块的共享内存大小
    vectorReduction<<<gridSize, blockSize, sharedMemoryBytes>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput, bytesOutput, cudaMemcpyDeviceToHost));
    // 对主机上的块结果进行最终归约
    float finalResult = 0.0f;
    for (const auto &val : hostOutput) {
        finalResult += val;
    }
    // 打印结果
    std::cout << "Final Reduction Result: " << finalResult << std::endl;
    // 释放内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    return 0;
}
