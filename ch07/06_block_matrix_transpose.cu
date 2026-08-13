// chapter7 片段 6（自 src/chapter7.txt 抽出）
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
// 核函数：使用共享内存进行块矩阵转置
__global__ void blockMatrixTranspose(float *input, float *output, int N) {
    // 定义共享内存，动态大小
    __shared__ float tile[32][32];
    // 全局索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 线程块内索引
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    // 加载数据到共享内存
    if (x < N && y < N) {
        tile[localY][localX] = input[y * N + x];
    }
    __syncthreads();
    // 转置后写回全局内存
    if (x < N && y < N) {
        output[x * N + y] = tile[localX][localY];
    }
}
int main() {
    const int N = 64; // 矩阵维度
    const size_t bytes = N * N * sizeof(float);
    // 主机内存分配并初始化
    float *hostInput = new float[N * N];
    float *hostOutput = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        hostInput[i] = static_cast<float>(i);
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
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 blockSize(32, 32); // 每个线程块32x32
    dim3 gridSize((N + 31) / 32, (N + 31) / 32);
    // 执行核函数
    blockMatrixTranspose<<<gridSize, blockSize>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    // 打印输出矩阵
    std::cout << "Output Matrix (Transposed):" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostOutput[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    delete[] hostInput;
    delete[] hostOutput;
    return 0;
}
