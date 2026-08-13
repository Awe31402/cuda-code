// chapter2 片段 1：全局索引计算 + 模拟工作负载 + 设备属性查询
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数：计算每个线程的全局索引
__global__ void computeGlobalIndex(int *output, int totalThreads) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < totalThreads) {
        output[threadId] = threadId; // 每个线程存储其全局索引
    }
}
// 核函数：简单计算，模拟工作负载
__global__ void performWorkload(int *data, int totalThreads) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < totalThreads) {
        for (int i = 0; i < 1000; ++i) { // 模拟计算工作负载
            data[threadId] += i;
        }
    }
}
void printDeviceProperties() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "设备名称: " << prop.name << std::endl;
    std::cout << "多处理器数量(SM): " << prop.multiProcessorCount << std::endl;
    std::cout << "每个SM最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个线程块最大线程数: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "每个线程块最大维度: (" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "每个网格最大维度: (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
}
int main() {
    // 查询设备属性
    printDeviceProperties();
    // 配置线程块和网格
    int totalThreads = 1024 * 32; // 假设需要计算的总线程数
    int threadsPerBlock = 256;    // 每个线程块的线程数
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "总线程数: " << totalThreads << std::endl;
    std::cout << "线程块大小: " << threadsPerBlock << std::endl;
    std::cout << "网格大小: " << numBlocks << std::endl;
    // 分配主机和设备内存
    int *hostOutput = new int[totalThreads];
    int *deviceOutput;
    cudaMalloc(&deviceOutput, totalThreads * sizeof(int));
    // 启动核函数
    auto start = std::chrono::high_resolution_clock::now();
    computeGlobalIndex<<<numBlocks, threadsPerBlock>>>(deviceOutput, totalThreads);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    // 拷贝数据回主机
    cudaMemcpy(hostOutput, deviceOutput, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);
    // 打印部分结果
    std::cout << "线程全局索引示例:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "线程 " << i << " 的全局索引: " << hostOutput[i] << std::endl;
    }
    // 性能测试
    std::cout << "计算全局索引耗时: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    // 模拟工作负载
    start = std::chrono::high_resolution_clock::now();
    performWorkload<<<numBlocks, threadsPerBlock>>>(deviceOutput, totalThreads);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "执行工作负载耗时: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    // 清理内存
    cudaFree(deviceOutput);
    delete[] hostOutput;
    return 0;
}
