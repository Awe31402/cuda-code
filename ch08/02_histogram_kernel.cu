// chapter8 片段 2（自 src/chapter8.txt 抽出）
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
// 核函数：使用原子加实现并行直方图
__global__ void histogramKernel(const int *data, int *histogram, int dataSize, int binCount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dataSize) {
        int bin = data[tid] % binCount; // 根据数据值计算对应的直方图桶
        atomicAdd(&histogram[bin], 1); // 使用原子加更新直方图桶
    }
}
int main() {
    const int dataSize = 1024; // 数据大小
    const int binCount = 10;  // 直方图桶数
    const size_t dataBytes = dataSize * sizeof(int);
    const size_t histBytes = binCount * sizeof(int);
    // 主机内存分配
    int *hostData = new int[dataSize];
    int *hostHistogram = new int[binCount]();
    // 初始化输入数据
    for (int i = 0; i < dataSize; ++i) {
        hostData[i] = rand() % 100; // 数据值范围为0-99
    }
    // 打印输入数据（部分展示）
    std::cout << "Input Data (Partial):" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << "... (Total: " << dataSize << " elements)" << std::endl;
    // 设备内存分配
    int *deviceData, *deviceHistogram;
    CUDA_CHECK(cudaMalloc(&deviceData, dataBytes));
    CUDA_CHECK(cudaMalloc(&deviceHistogram, histBytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceData, hostData, dataBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceHistogram, 0, histBytes));
    // 配置线程块和网格
    int threadsPerBlock = 256;
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    // 执行核函数
    histogramKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceData, deviceHistogram, dataSize, binCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 将直方图结果传回主机
    CUDA_CHECK(cudaMemcpy(hostHistogram, deviceHistogram, histBytes, cudaMemcpyDeviceToHost));
    // 打印直方图结果
    std::cout << "Histogram Result:" << std::endl;
    for (int i = 0; i < binCount; ++i) {
        std::cout << "Bin " << i << ": " << hostHistogram[i] << std::endl;
    }
    // 清理内存
    CUDA_CHECK(cudaFree(deviceData));
    CUDA_CHECK(cudaFree(deviceHistogram));
    delete[] hostData;
    delete[] hostHistogram;
    return 0;
}
