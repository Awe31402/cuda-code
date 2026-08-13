// chapter4 片段 3（自 src/chapter4.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define DATA_SIZE 1024 * 1024 * 10 // 数据大小
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << " 错误: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    const size_t bytes = DATA_SIZE * sizeof(float);
    // 分配分页内存
    float *hostPageable = new float[DATA_SIZE];
    // 分配锁页内存
    float *hostPinned;
    cudaHostAlloc(&hostPinned, bytes, cudaHostAllocDefault);
    // 初始化数据
    for (int i = 0; i < DATA_SIZE; ++i) {
        hostPageable[i] = static_cast<float>(i);
        hostPinned[i] = static_cast<float>(i);
    }
    // 分配设备内存
    float *deviceData;
    cudaMalloc(&deviceData, bytes);
    checkCudaError("设备内存分配失败");
    // 测试分页内存数据传输
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(deviceData, hostPageable, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto pageableDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 测试锁页内存数据传输
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(deviceData, hostPinned, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto pinnedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 打印结果
    std::cout << "分页内存传输时间: " << pageableDuration << " ms" << std::endl;
    std::cout << "锁页内存传输时间: " << pinnedDuration << " ms" << std::endl;
    // 清理资源
    cudaFree(deviceData);
    cudaFreeHost(hostPinned);
    delete[] hostPageable;
    return 0;
}
