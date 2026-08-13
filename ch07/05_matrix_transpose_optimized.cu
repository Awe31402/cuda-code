// chapter7 片段 5（自 src/chapter7.txt 抽出）
// 修正：原文此段沒有 include 也沒有 CUDA_CHECK 定義（沿用前一段的環境），
// 這裡補上與同章其他段落一致的前置宣告。
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
__global__ void matrixTransposeOptimized(float *input, float *output, int N) {
    __shared__ float sharedMemory[32][33]; // 添加偏移量
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        sharedMemory[threadIdx.y][threadIdx.x] = input[y * N + x];
        __syncthreads();
        output[x * N + y] = sharedMemory[threadIdx.y][threadIdx.x];
    }
}
int main() {
    const int N = 32;
    const size_t bytes = N * N * sizeof(float);
    // 主机内存分配并初始化
    float *hostInput = new float[N * N];
    float *hostOutput = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        hostInput[i] = static_cast<float>(i);
    }
    // 设备内存分配
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 blockSize(32, 32);
    dim3 gridSize((N + 31) / 32, (N + 31) / 32);
    // 执行优化后的核函数
    matrixTransposeOptimized<<<gridSize, blockSize>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    // 打印结果
    std::cout << "Output without Bank Conflict (First 10 Elements):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    delete[] hostInput;
    delete[] hostOutput;
    return 0;
}
