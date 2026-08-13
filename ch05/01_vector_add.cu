// chapter5 片段 1（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 宏定义：通用错误检测函数
#define CHECK_CUDA_ERROR(call)                                                     \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess) {                                                  \
            std::cerr << "CUDA错误: " << cudaGetErrorString(err)                   \
                      << " 在文件 " << __FILE__                                    \
                      << " 的第 " << __LINE__ << " 行" << std::endl;              \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    }
// 核函数：简单向量加法
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);
    // 主机内存分配
    float *hostA = new float[N];
    float *hostB = new float[N];
    float *hostC = new float[N];
    // 初始化主机数据
    for (int i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceA, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&deviceB, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&deviceC, bytes));
    // 数据传输：主机到设备
    CHECK_CUDA_ERROR(cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice));
    // 启动核函数
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, N);
    // 检查核函数执行错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // 数据传输：设备到主机
    CHECK_CUDA_ERROR(cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost));
    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (hostC[i] != hostA[i] + hostB[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    // 清理资源
    CHECK_CUDA_ERROR(cudaFree(deviceA));
    CHECK_CUDA_ERROR(cudaFree(deviceB));
    CHECK_CUDA_ERROR(cudaFree(deviceC));
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
