// chapter4 片段 1（自 src/chapter4.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define DATA_SIZE 1024 * 1024 // 数据大小
// 核函数：无循环展开的数组求和
__global__ void arraySumBasic(const float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + 1.0f; // 简单的数组加操作
    }
}
// 核函数：使用循环展开优化的数组求和
__global__ void arraySumUnrolled(const float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每个线程处理多项数据
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        output[i] = input[i] + 1.0f;
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
    const int bytes = DATA_SIZE * sizeof(float);
    // 分配主机内存
    float *hostInput = new float[DATA_SIZE];
    float *hostOutputBasic = new float[DATA_SIZE];
    float *hostOutputUnrolled = new float[DATA_SIZE];
    // 初始化输入数据
    for (int i = 0; i < DATA_SIZE; ++i) {
        hostInput[i] = static_cast<float>(i);
    }
    // 分配设备内存
    float *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, bytes);
    cudaMalloc(&deviceOutput, bytes);
    checkCudaError("设备内存分配失败");
    // 主机到设备数据传输
    cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    dim3 blockDim(256); // 每个线程块256个线程
    dim3 gridDim((DATA_SIZE + blockDim.x - 1) / blockDim.x);
    // 基础核函数执行
    auto start = std::chrono::high_resolution_clock::now();
    arraySumBasic<<<gridDim, blockDim>>>(deviceInput, deviceOutput, DATA_SIZE);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto basicDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 从设备到主机数据传输
    cudaMemcpy(hostOutputBasic, deviceOutput, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 使用循环展开的核函数执行
    start = std::chrono::high_resolution_clock::now();
    arraySumUnrolled<<<gridDim, blockDim>>>(deviceInput, deviceOutput, DATA_SIZE);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto unrolledDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 从设备到主机数据传输
    cudaMemcpy(hostOutputUnrolled, deviceOutput, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 验证结果
    bool correct = true;
    for (int i = 0; i < DATA_SIZE; ++i) {
        if (hostOutputBasic[i] != hostOutputUnrolled[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "基础版本执行时间: " << basicDuration << " ms" << std::endl;
    std::cout << "循环展开版本执行时间: " << unrolledDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    // 释放资源
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutputBasic;
    delete[] hostOutputUnrolled;
    return 0;
}
