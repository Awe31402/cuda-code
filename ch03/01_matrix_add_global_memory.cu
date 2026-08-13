// chapter3 片段 1（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 矩阵加法核函数，使用全局内存
__global__ void matrixAddGlobalMemory(const float *a, const float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < N && idy < N) {
        int index = idy * N + idx;
        c[index] = a[index] + b[index];
    }
}
// 矩阵加法核函数，使用共享内存
__global__ void matrixAddSharedMemory(const float *a, const float *b, float *c, int N) {
    __shared__ float tileA[32][32]; // 分配共享内存
    __shared__ float tileB[32][32];
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < N && idy < N) {
        // 将全局内存数据加载到共享内存
        int index = idy * N + idx;
        tileA[ty][tx] = a[index];
        tileB[ty][tx] = b[index];
        __syncthreads(); // 确保共享内存加载完成
        // 执行矩阵加法
        c[index] = tileA[ty][tx] + tileB[ty][tx];
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
    const int N = 1024; // 矩阵大小 N x N
    const int size = N * N;
    const int bytes = size * sizeof(float);
    float *hostA = new float[size];
    float *hostB = new float[size];
    float *hostC = new float[size];
    // 初始化矩阵数据
    for (int i = 0; i < size; ++i) {
        hostA[i] = 1.0f;
        hostB[i] = 2.0f;
    }
    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    checkCudaError("设备内存分配失败");
    // 拷贝数据到设备
    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    // 配置线程块和网格
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    // 全局内存版本计时
    auto start = std::chrono::high_resolution_clock::now();
    matrixAddGlobalMemory<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto globalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // 拷贝结果回主机
    cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    std::cout << "全局内存版本执行时间: " << globalDuration << " ms" << std::endl;
    // 共享内存版本计时
    start = std::chrono::high_resolution_clock::now();
    matrixAddSharedMemory<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto sharedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "共享内存版本执行时间: " << sharedDuration << " ms" << std::endl;
    // 清理资源
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
