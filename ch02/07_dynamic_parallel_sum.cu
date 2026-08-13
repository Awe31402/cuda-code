// chapter2 片段 7：动态并行递归求和（原文代码，需 -rdc=true）
// 已知问题：设备端 cudaDeviceSynchronize() 在 CUDA 11.6 起弃用、CUDA 12 起移除，
// 因此本文件在 CUDA 12 下无法编译，仅作原文存档，可运行版见 07_dynamic_parallel_sum_fixed.cu
#include <cuda_runtime.h>
#include <iostream>
// 动态并行核函数：递归计算数组元素平方和
__global__ void recursiveSum(int *data, int size, int *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (size <= 1) {
        // 基本情况：数组只有一个元素
        if (idx == 0) {
            *result = data[0]; // 返回最终结果
        }
        return;
    }
    if (idx < size / 2) {
        // 每个线程计算两个元素的平方和
        data[idx] += data[idx + size / 2];
    }
    __syncthreads(); // 确保所有线程完成操作
    if (threadIdx.x == 0) {
        // 递归调用新网格
        int newSize = size / 2;
        recursiveSum<<<1, newSize>>>(data, newSize, result);
        cudaDeviceSynchronize();
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
    const int dataSize = 16; // 数组大小
    int hostData[dataSize]; // 主机数据
    int hostResult;         // 最终结果
    // 初始化数组
    for (int i = 0; i < dataSize; ++i) {
        hostData[i] = i + 1; // 数据为1, 2, ..., 16
    }
    // 分配设备内存
    int *deviceData, *deviceResult;
    cudaMalloc(&deviceData, dataSize * sizeof(int));
    cudaMalloc(&deviceResult, sizeof(int));
    checkCudaError("设备内存分配失败");
    // 拷贝数据到设备
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    // 启动核函数
    recursiveSum<<<1, dataSize>>>(deviceData, dataSize, deviceResult);
    cudaDeviceSynchronize();
    checkCudaError("核函数执行失败");
    // 拷贝结果回主机
    cudaMemcpy(&hostResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 输出结果
    std::cout << "递归计算数组平方和结果: " << hostResult << std::endl;
    // 释放内存
    cudaFree(deviceData);
    cudaFree(deviceResult);
    return 0;
}
