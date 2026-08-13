// chapter8 片段 5（自 src/chapter8.txt 抽出）
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
// 核函数：使用Shuffle指令实现Warp内归约
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset); // 邻近线程间交换数据
    }
    return val;
}
// 核函数：线程块内的归约计算
__global__ void reduceKernel(const int *input, int *output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % 32; // Warp内线程索引
    int warpID = threadIdx.x / 32; // Warp编号
    __shared__ int sharedData[32]; // 用于存储每个Warp的归约结果
    // 每个线程加载一个数据
    int value = (tid < N) ? input[tid] : 0;
    // 使用Warp Shuffle进行归约
    int warpSum = warpReduceSum(value);
    // 每个Warp的线程0将结果写入共享内存
    if (lane == 0) {
        sharedData[warpID] = warpSum;
    }
    __syncthreads(); // 确保所有Warp结果已写入共享内存
    // 线程块内线程0归约所有Warp的结果
    if (threadIdx.x == 0) {
        int blockSum = 0;
        for (int i = 0; i < (blockDim.x + 31) / 32; ++i) {
            blockSum += sharedData[i];
        }
        atomicAdd(output, blockSum); // 全局原子加
    }
}
int main() {
    const int N = 1024; // 输入数组大小
    const size_t bytes = N * sizeof(int);
    // 主机内存分配
    int *hostInput = new int[N];
    int hostOutput = 0;
    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        hostInput[i] = 1; // 每个元素初始化为1
    }
    // 设备内存分配
    int *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, sizeof(int)));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceOutput, 0, sizeof(int)));
    // 配置线程块和网格
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // 执行核函数
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 将结果传回主机
    CUDA_CHECK(cudaMemcpy(&hostOutput, deviceOutput, sizeof(int), cudaMemcpyDeviceToHost));
    // 打印结果
    std::cout << "Final Sum: " << hostOutput << std::endl;
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    delete[] hostInput;
    return 0;
}
