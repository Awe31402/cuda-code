// chapter3 片段 3（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define TILE_SIZE 32 // 共享内存块大小
// 未优化的矩阵转置
__global__ void matrixTransposeNaive(const float *input, float *output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        output[x * N + y] = input[y * N + x]; // 非合并访问
    }
}
// 优化的矩阵转置，利用共享内存
__global__ void matrixTransposeOptimized(const float *input, float *output, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // 避免Bank冲突
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (x < N && y < N) {
        tile[ty][tx] = input[y * N + x]; // 合并访问加载到共享内存
    }
    __syncthreads();
    int transposedX = blockIdx.y * blockDim.y + threadIdx.x;
    int transposedY = blockIdx.x * blockDim.x + threadIdx.y;
    if (transposedX < N && transposedY < N) {
        output[transposedY * N + transposedX] = tile[tx][ty]; // 合并访问存储
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
    float *hostInput = new float[size];
    float *hostOutputNaive = new float[size];
    float *hostOutputOptimized = new float[size];
    // 初始化输入矩阵
    for (int i = 0; i < size; ++i) {
        hostInput[i] = static_cast<float>(i);
    }
    // 分配设备内存
    float *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, bytes);
    cudaMalloc(&deviceOutput, bytes);
    checkCudaError("设备内存分配失败");
    // 拷贝数据到设备
    cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    // 配置线程块和网格
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    // 未优化版本计时
    auto start = std::chrono::high_resolution_clock::now();
    matrixTransposeNaive<<<gridDim, blockDim>>>(deviceInput, deviceOutput, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto naiveDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostOutputNaive, deviceOutput, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    std::cout << "未优化版本执行时间: " << naiveDuration << " ms" << std::endl;
    // 优化版本计时
    start = std::chrono::high_resolution_clock::now();
    matrixTransposeOptimized<<<gridDim, blockDim>>>(deviceInput, deviceOutput, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto optimizedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostOutputOptimized, deviceOutput, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    std::cout << "优化版本执行时间: " << optimizedDuration << " ms" << std::endl;
    // 验证结果一致性
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (hostOutputNaive[i] != hostOutputOptimized[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    // 清理资源
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutputNaive;
    delete[] hostOutputOptimized;
    return 0;
}
