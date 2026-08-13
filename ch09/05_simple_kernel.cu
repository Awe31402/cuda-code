// chapter9 片段 5（自 src/chapter9.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 定义矩阵大小
#define N 256
// 核函数模拟任务
__global__ void simpleKernel(float* data, int stream_id) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] += stream_id;  // 模拟不同流的操作
    }
}
int main() {
    // 定义变量
    float* d_data1;
    float* d_data2;
    float h_data1[N];
    float h_data2[N];
    // 初始化主机数据
    for (int i = 0; i < N; i++) {
        h_data1[i] = 0.0f;
        h_data2[i] = 0.0f;
    }
    // 设备内存分配
    cudaMalloc((void**)&d_data1, N * sizeof(float));
    cudaMalloc((void**)&d_data2, N * sizeof(float));
    // 创建两个不同优先级的流
    cudaStream_t high_priority_stream, low_priority_stream;
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&high_priority_stream, cudaStreamNonBlocking, greatestPriority);
    cudaStreamCreateWithPriority(&low_priority_stream, cudaStreamNonBlocking, leastPriority);
    // 数据传输到设备
    cudaMemcpyAsync(d_data1, h_data1, N * sizeof(float), cudaMemcpyHostToDevice, high_priority_stream);
    cudaMemcpyAsync(d_data2, h_data2, N * sizeof(float), cudaMemcpyHostToDevice, low_priority_stream);
    // 启动核函数
    simpleKernel<<<(N + 255) / 256, 256, 0, high_priority_stream>>>(d_data1, 1);
    simpleKernel<<<(N + 255) / 256, 256, 0, low_priority_stream>>>(d_data2, 2);
    // 结果传回主机
    cudaMemcpyAsync(h_data1, d_data1, N * sizeof(float), cudaMemcpyDeviceToHost, high_priority_stream);
    cudaMemcpyAsync(h_data2, d_data2, N * sizeof(float), cudaMemcpyDeviceToHost, low_priority_stream);
    // 等待流完成
    cudaStreamSynchronize(high_priority_stream);
    cudaStreamSynchronize(low_priority_stream);
    // 打印部分结果
    std::cout << "High priority stream results:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data1[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Low priority stream results:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data2[i] << " ";
    }
    std::cout << "\n";
    // 清理资源
    cudaStreamDestroy(high_priority_stream);
    cudaStreamDestroy(low_priority_stream);
    cudaFree(d_data1);
    cudaFree(d_data2);
    return 0;
}
