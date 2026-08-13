// chapter8 片段 7（自 src/chapter8.txt 抽出）
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <cstdlib>
namespace cg = cooperative_groups;
#define CUDA_CHECK(call)                                                                 \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }
// 核函数：利用协作组实现块内归约
__global__ void blockReduceSum(const int *input, int *output, int N) {
    // 定义线程块范围的协作组
    cg::thread_block block = cg::this_thread_block();
    // 使用共享内存存储中间结果
    __shared__ int sharedData[1024];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 加载数据到共享内存
    int value = (tid < N) ? input[tid] : 0;
    sharedData[threadIdx.x] = value;
    // 同步线程块内所有线程，确保共享内存加载完成
    block.sync();
    // 归约操作：线程块范围
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        }
        block.sync(); // 每次归约后同步
    }
    // 将结果写入全局内存
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
int main() {
    const int N = 1 << 20; // 数据大小（1百万个元素）
    const size_t bytes = N * sizeof(int);
    // 主机内存分配
    int *hostInput = new int[N];
    int *hostOutput;
    int hostFinalResult = 0;
    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        hostInput[i] = 1; // 初始化为1，方便验证结果
    }
    // 设备内存分配
    int *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, sizeof(int) * 1024)); // 每个块的结果
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    // 配置线程块和网格
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // 执行核函数
    blockReduceSum<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 分配主机内存存储中间结果
    hostOutput = new int[blocksPerGrid];
    // 将中间结果传回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost));
    // 在主机完成最终归约
    for (int i = 0; i < blocksPerGrid; ++i) {
        hostFinalResult += hostOutput[i];
    }
    // 打印结果
    std::cout << "Final Sum: " << hostFinalResult << std::endl;
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    delete[] hostInput;
    delete[] hostOutput;
    return 0;
}
