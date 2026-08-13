// chapter2 片段 9：分支发散 vs 分支规约
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数：存在分支发散
__global__ void branchDivergence(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        if (idx % 2 == 0) {
            data[idx] *= 2; // 偶数元素乘以2
        } else {
            data[idx] += 1; // 奇数元素加1
        }
    }
}
// 核函数：通过分支规约优化
__global__ void branchReduction(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int value = data[idx];
        data[idx] = (idx % 2 == 0) ? value * 2 : value + 1; // 使用规约合并分支
    }
}
// 检查CUDA错误
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << " 错误: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    const int dataSize = 1 << 20; // 数据量 (1M元素)
    int *hostData = new int[dataSize];
    int *deviceData;
    // 初始化主机数据
    for (int i = 0; i < dataSize; ++i) {
        hostData[i] = i;
    }
    // 分配设备内存
    cudaMalloc(&deviceData, dataSize * sizeof(int));
    checkCudaError("设备内存分配失败");
    // 将数据从主机拷贝到设备
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    // 测试分支发散
    dim3 blockDim(256);
    dim3 gridDim((dataSize + blockDim.x - 1) / blockDim.x);
    auto start = std::chrono::high_resolution_clock::now();
    branchDivergence<<<gridDim, blockDim>>>(deviceData, dataSize);
    cudaDeviceSynchronize();
    checkCudaError("分支发散核函数执行失败");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "分支发散核函数执行时间: " << duration << " ms" << std::endl;
    // 测试分支规约
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    start = std::chrono::high_resolution_clock::now();
    branchReduction<<<gridDim, blockDim>>>(deviceData, dataSize);
    cudaDeviceSynchronize();
    checkCudaError("分支规约核函数执行失败");
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "分支规约核函数执行时间: " << duration << " ms" << std::endl;
    // 将结果从设备拷贝回主机
    cudaMemcpy(hostData, deviceData, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 输出部分结果
    std::cout << "结果验证:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "hostData[" << i << "] = " << hostData[i] << std::endl;
    }
    // 释放内存
    cudaFree(deviceData);
    delete[] hostData;
    return 0;
}
