// chapter4 片段 6（自 src/chapter4.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define N 512 // 矩阵大小
// 核函数：矩阵加法
__global__ void matrixAdd(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << msg << " 错误: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
int main() {
    const int size = N * N;
    const int bytes = size * sizeof(float);
    // 分配Unified Memory
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    checkCudaError("Unified Memory分配失败");
    // 初始化数据
    for (int i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    // 设置内存访问建议
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, 0); // GPU 0
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, 1); // GPU 1
    // 确定使用的设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        std::cerr << "需要至少两个GPU设备" << std::endl;
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        return 1;
    }
    // 分别在两个设备上执行矩阵加法
    cudaSetDevice(0);
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixAdd<<<gridDim, blockDim>>>(a, b, c, N);
    cudaDeviceSynchronize();
    checkCudaError("GPU 0核函数执行失败");
    cudaSetDevice(1);
    matrixAdd<<<gridDim, blockDim>>>(a, b, c, N);
    cudaDeviceSynchronize();
    checkCudaError("GPU 1核函数执行失败");
    // 验证结果
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (c[i] != a[i] + b[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    // 清理资源
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
