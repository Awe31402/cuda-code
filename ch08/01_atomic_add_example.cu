// chapter8 片段 1（自 src/chapter8.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#define CUDA_CHECK(call)                                                                 \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }
// 核函数：使用原子加实现数组求和
__global__ void atomicAddExample(const int *input, int *result, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        atomicAdd(result, input[tid]);
    }
}
int main() {
    const int N = 1024; // 数组大小
    const size_t bytes = N * sizeof(int);
    // 主机内存分配
    int *hostInput = new int[N];
    int hostResult = 0;
    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        hostInput[i] = 1; // 每个元素初始化为1
    }
    // 打印输入数据
    std::cout << "Input Array:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << hostInput[i] << " ";
    }
    std::cout << std::endl;
    // 设备内存分配
    int *deviceInput, *deviceResult;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceResult, sizeof(int)));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceResult, 0, sizeof(int)));
    // 配置线程块和网格
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // 执行核函数
    atomicAddExample<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceResult, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 结果传回主机
    CUDA_CHECK(cudaMemcpy(&hostResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost));
    // 打印结果
    std::cout << "Result of atomicAdd: " << hostResult << std::endl;
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceResult));
    delete[] hostInput;
    return 0;
}
