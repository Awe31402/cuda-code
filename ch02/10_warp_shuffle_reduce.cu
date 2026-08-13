// chapter2 片段 10：Warp Shuffle 归约求和
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数：使用Warp Shuffle实现Warp内归约求和
__global__ void warpReduceSum(int *input, int *output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % warpSize; // 线程在Warp内的索引
    int warpId = threadIdx.x / warpSize; // 当前线程所属的Warp编号
    (void)warpId;
    // 初始化归约值
    int value = (idx < N) ? input[idx] : 0;
    // 使用Warp Shuffle实现归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    // 将每个Warp的结果写入共享内存
    if (lane == 0) {
        atomicAdd(output, value); // 原子操作，避免冲突
    }
}
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << " 错误: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    const int dataSize = 1 << 20; // 数据量 (1M元素)
    int *hostInput = new int[dataSize];
    int hostOutput = 0;
    // 初始化输入数据
    for (int i = 0; i < dataSize; ++i) {
        hostInput[i] = 1; // 每个元素初始化为1
    }
    // 分配设备内存
    int *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, dataSize * sizeof(int));
    cudaMalloc(&deviceOutput, sizeof(int));
    checkCudaError("设备内存分配失败");
    // 拷贝数据到设备
    cudaMemcpy(deviceInput, hostInput, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(deviceOutput, 0, sizeof(int)); // 初始化输出为0
    checkCudaError("主机到设备数据传输失败");
    // 配置线程块和网格
    dim3 blockDim(256);
    dim3 gridDim((dataSize + blockDim.x - 1) / blockDim.x);
    // 启动核函数
    auto start = std::chrono::high_resolution_clock::now();
    warpReduceSum<<<gridDim, blockDim>>>(deviceInput, deviceOutput, dataSize);
    cudaDeviceSynchronize();
    checkCudaError("核函数执行失败");
    auto end = std::chrono::high_resolution_clock::now();
    // 拷贝结果回主机
    cudaMemcpy(&hostOutput, deviceOutput, sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 输出结果
    std::cout << "数组总和: " << hostOutput << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "执行时间: " << duration << " ms" << std::endl;
    // 释放内存
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    return 0;
}
