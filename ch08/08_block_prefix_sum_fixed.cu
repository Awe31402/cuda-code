// chapter8 片段 8（修正版）：分塊前綴和
//
// 原文問題：抽出區塊和（block sums）時寫成
//     blockPrefixSum<<<1, gridSize, ...>>>(deviceOutput + blockSize - 1, blockSums, gridSize);
// 這只是從 deviceOutput[1023] 開始「連續」讀 gridSize 個元素，
// 但每個區塊的總和其實散落在 index 1023, 2047, 3071, ...（間隔 blockSize）。
// 因此 blockSums 全錯，原文執行結果會印出 "Error in prefix sum computation."
//
// 修正：加一個 gatherBlockSums 核函數，用 blockSize 當 stride 把各區塊最後一個元素收集起來。
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
// 核函数：线程块范围内的前缀和（inclusive scan）
__global__ void blockPrefixSum(int *input, int *output, int N) {
    extern __shared__ int sharedData[]; // 动态共享内存
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    // 将数据加载到共享内存
    if (tid < N) {
        sharedData[tx] = input[tid];
    } else {
        sharedData[tx] = 0;
    }
    __syncthreads();
    // 前缀和计算
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (tx >= stride) {
            temp = sharedData[tx - stride];
        }
        __syncthreads(); // 确保读取完成
        sharedData[tx] += temp;
        __syncthreads(); // 确保写入完成
    }
    // 写入输出
    if (tid < N) {
        output[tid] = sharedData[tx];
    }
}
// 新增核函数：按 blockSize 为步长收集每个线程块的总和
__global__ void gatherBlockSums(const int *output, int *blockSums, int numBlocks, int blockSize) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < numBlocks) {
        blockSums[b] = output[(b + 1) * blockSize - 1];
    }
}
// 核函数：调整线程块结果
__global__ void adjustBlocks(int *output, int *blockSums, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && tid < N) {
        output[tid] += blockSums[blockIdx.x - 1];
    }
}
// 主函数
int main() {
    const int N = 1 << 20; // 数据大小（1百万个元素）
    const int blockSize = 1024; // 每个线程块的线程数
    const int gridSize = (N + blockSize - 1) / blockSize; // 网格大小
    size_t bytes = N * sizeof(int);
    size_t blockSumBytes = gridSize * sizeof(int);
    // 主机内存分配
    int *hostInput = new int[N];
    int *hostOutput = new int[N];
    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        hostInput[i] = 1; // 每个元素为1，便于验证
    }
    // 设备内存分配
    int *deviceInput, *deviceOutput, *blockSums;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    CUDA_CHECK(cudaMalloc(&blockSums, blockSumBytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    // 1. 各线程块内部的前缀和
    blockPrefixSum<<<gridSize, blockSize, blockSize * sizeof(int)>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 2. 以 blockSize 为步长收集每块总和（原文缺这一步）
    gatherBlockSums<<<(gridSize + 255) / 256, 256>>>(deviceOutput, blockSums, gridSize, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 3. 对块总和本身做前缀和（gridSize == 1024，单个线程块即可完成）
    blockPrefixSum<<<1, gridSize, gridSize * sizeof(int)>>>(blockSums, blockSums, gridSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 4. 调整线程块结果
    adjustBlocks<<<gridSize, blockSize>>>(deviceOutput, blockSums, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 将结果传回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    // 验证结果
    bool success = true;
    int badIndex = -1;
    for (int i = 0; i < N; ++i) {
        if (hostOutput[i] != i + 1) {
            success = false;
            badIndex = i;
            break;
        }
    }
    if (success) {
        std::cout << "Prefix sum computed successfully. (last = "
                  << hostOutput[N - 1] << ", 期望 " << N << ")" << std::endl;
    } else {
        std::cout << "Error in prefix sum computation. 第一個錯誤在 index "
                  << badIndex << "，值 " << hostOutput[badIndex]
                  << "，期望 " << badIndex + 1 << std::endl;
    }
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(blockSums));
    delete[] hostInput;
    delete[] hostOutput;
    return success ? 0 : 1;
}
