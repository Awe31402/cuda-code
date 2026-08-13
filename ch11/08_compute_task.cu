// chapter11 片段 8（自 src/chapter11.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
// 核函数模拟不同任务
__global__ void computeTask(int *data, int size, int factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = data[idx] * factor;
    }
}
void resourceSchedulingOptimization(int *h_data, int size, int num_tasks, int num_streams) {
    int *d_data;
    cudaStream_t *streams = new cudaStream_t[num_streams];
    cudaEvent_t start, stop;
    // 分配设备内存
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
    // 创建流和事件
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start, 0);
    // 动态分配任务到流
    int chunk_size = size / num_tasks;
    for (int i = 0; i < num_tasks; ++i) {
        int offset = i * chunk_size;
        int current_size = (i == num_tasks - 1) ? size - offset : chunk_size;
        computeTask<<<(current_size + 255) / 256, 256, 0, streams[i % num_streams]>>>(d_data + offset, current_size, i + 1);
    }
    // 同步所有流
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // 计算耗时
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;
    // 拷回结果
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    // 释放资源
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] streams;
}
int main() {
    const int size = 1000000;
    const int num_tasks = 10;
    const int num_streams = 4;
    int *h_data = new int[size];
    for (int i = 0; i < size; ++i) {
        h_data[i] = i % 100;
    }
    auto start = std::chrono::high_resolution_clock::now();
    resourceSchedulingOptimization(h_data, size, num_tasks, num_streams);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    delete[] h_data;
    return 0;
}
