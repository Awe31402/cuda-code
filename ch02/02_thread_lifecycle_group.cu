// chapter2 片段 2：线程生命周期（分支）与 Warp 分组
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数1: 计算线程全局索引并模拟不同分支
__global__ void computeThreadLifecycle(int *globalIndex, int *branchResult, int totalThreads) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x; // 计算全局索引
    if (threadId < totalThreads) {
        globalIndex[threadId] = threadId; // 保存全局索引
        // 模拟条件分支，判断线程ID的奇偶性
        if (threadId % 2 == 0) {
            branchResult[threadId] = threadId * 2; // 偶数分支
        } else {
            branchResult[threadId] = threadId * 3; // 奇数分支
        }
    }
}
// 核函数2: 线程分组操作
__global__ void groupThreads(int *groupedData, int totalThreads) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x; // 全局索引
    if (threadId < totalThreads) {
        int groupId = threadIdx.x / 32; // 以Warp为单位进行分组
        groupedData[threadId] = groupId; // 保存每个线程的组ID
    }
}
void printDeviceProperties() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "设备名称: " << prop.name << std::endl;
    std::cout << "流式多处理器数量(SM): " << prop.multiProcessorCount << std::endl;
    std::cout << "每个线程块最大线程数: " << prop.maxThreadsPerBlock << std::endl;
}
int main() {
    // 查询设备属性
    printDeviceProperties();
    // 配置线程和线程块
    int totalThreads = 1024;       // 总线程数
    int threadsPerBlock = 256;    // 每个线程块的线程数
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "总线程数: " << totalThreads << std::endl;
    std::cout << "线程块大小: " << threadsPerBlock << std::endl;
    std::cout << "网格大小: " << numBlocks << std::endl;
    // 分配主机和设备内存
    int *hostGlobalIndex = new int[totalThreads];
    int *hostBranchResult = new int[totalThreads];
    int *hostGroupedData = new int[totalThreads];
    int *deviceGlobalIndex, *deviceBranchResult, *deviceGroupedData;
    cudaMalloc(&deviceGlobalIndex, totalThreads * sizeof(int));
    cudaMalloc(&deviceBranchResult, totalThreads * sizeof(int));
    cudaMalloc(&deviceGroupedData, totalThreads * sizeof(int));
    // 启动核函数1: 线程生命周期模拟
    auto start = std::chrono::high_resolution_clock::now();
    computeThreadLifecycle<<<numBlocks, threadsPerBlock>>>(deviceGlobalIndex, deviceBranchResult, totalThreads);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    // 拷贝数据回主机
    cudaMemcpy(hostGlobalIndex, deviceGlobalIndex, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostBranchResult, deviceBranchResult, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "线程生命周期模拟完成，执行时间: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    // 打印部分结果
    std::cout << "线程索引和分支结果示例:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "线程 " << i << " 的全局索引: " << hostGlobalIndex[i]
                  << ", 分支结果: " << hostBranchResult[i] << std::endl;
    }
    // 启动核函数2: 分组操作
    start = std::chrono::high_resolution_clock::now();
    groupThreads<<<numBlocks, threadsPerBlock>>>(deviceGroupedData, totalThreads);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(hostGroupedData, deviceGroupedData, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "线程分组完成，执行时间: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    // 打印部分分组结果
    std::cout << "线程分组示例:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "线程 " << i << " 所属分组: " << hostGroupedData[i] << std::endl;
    }
    // 清理内存
    cudaFree(deviceGlobalIndex);
    cudaFree(deviceBranchResult);
    cudaFree(deviceGroupedData);
    delete[] hostGlobalIndex;
    delete[] hostBranchResult;
    delete[] hostGroupedData;
    return 0;
}
