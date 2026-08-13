// chapter7 片段 7（自 src/chapter7.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(err) << std::endl;                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }
// 核函数：使用共享内存进行矩阵转置和求和
__global__ void matrixTransposeAndSum(float *input, float *output, float *sum, int N) {
    __shared__ float tile[32][33]; // 添加偏移量避免Bank冲突
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    // 初始化求和的共享变量
    __shared__ float blockSum;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        blockSum = 0.0f;
    }
    __syncthreads();
    // 加载数据到共享内存并计算块内求和
    if (x < N && y < N) {
        float value = input[y * N + x];
        tile[localY][localX] = value;
        atomicAdd(&blockSum, value); // 原子操作进行块内求和
    }
    __syncthreads();
    // 写回转置数据
    if (x < N && y < N) {
        output[x * N + y] = tile[localX][localY];
    }
    // 累加块内求和结果到全局内存
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(sum, blockSum);
    }
}
int main() {
    const int N = 64; // 矩阵维度
    const size_t bytes = N * N * sizeof(float);
    // 主机内存分配并初始化
    float *hostInput = new float[N * N];
    float *hostOutput = new float[N * N];
    float hostSum = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        hostInput[i] = static_cast<float>(i + 1);
    }
    // 打印输入矩阵
    std::cout << "Input Matrix:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostInput[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    // 设备内存分配
    float *deviceInput, *deviceOutput, *deviceSum;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceSum, sizeof(float)));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceSum, &hostSum, sizeof(float), cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 blockSize(32, 32); // 每个线程块32x32
    dim3 gridSize((N + 31) / 32, (N + 31) / 32);
    // 执行核函数
    matrixTransposeAndSum<<<gridSize, blockSize>>>(deviceInput, deviceOutput, deviceSum, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hostSum, deviceSum, sizeof(float), cudaMemcpyDeviceToHost));
    // 打印转置后的矩阵
    std::cout << "Output Matrix (Transposed):" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostOutput[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    // 打印矩阵总和
    std::cout << "Sum of Matrix Elements: " << hostSum << std::endl;
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(deviceSum));
    delete[] hostInput;
    delete[] hostOutput;
    return 0;
}
