// chapter5 片段 3（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 核函数：简单矩阵加法
__global__ void matrixAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n * n) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    const int N = 10; // 矩阵大小为10x10
    const int size = N * N * sizeof(float);
    // 分配主机内存
    float *hostA = new float[N * N];
    float *hostB = new float[N * N];
    float *hostC = new float[N * N]; // 故意不初始化hostC
    for (int i = 0; i < N * N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, size);
    cudaMalloc(&deviceB, size);
    cudaMalloc(&deviceC, size);
    // 将数据从主机传输到设备
    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);
    // 启动核函数
    dim3 blockSize(256);
    dim3 gridSize((N * N + blockSize.x - 1) / blockSize.x);
    matrixAdd<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, N);
    // 从设备传输结果到主机
    cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);
    // 故意访问未分配的内存（模拟内存越界）
    std::cout << "模拟内存越界访问: " << hostC[N * N + 1] << std::endl;
    // 释放设备内存
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    // 释放主机内存
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
