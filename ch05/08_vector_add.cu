// chapter5 片段 8（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 核函数：简单向量加法
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    const int N = 1 << 20; // 向量大小
    const int bytes = N * sizeof(float);
    // 分配主机内存并初始化
    float *hostA = new float[N];
    float *hostB = new float[N];
    float *hostC = new float[N];
    for (int i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(2 * i);
    }
    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    // 创建CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    // 异步数据传输到设备
    cudaMemcpyAsync(deviceA, hostA, bytes, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceB, hostB, bytes, cudaMemcpyHostToDevice, stream2);
    // 启动核函数
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize, 0, stream1>>>(deviceA, deviceB, deviceC, N);
    // 异步将结果传输回主机
    cudaMemcpyAsync(hostC, deviceC, bytes, cudaMemcpyDeviceToHost, stream2);
    // 同步流
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "]: " << hostC[i] << std::endl;
    }
    // 销毁流
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    // 释放内存
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
