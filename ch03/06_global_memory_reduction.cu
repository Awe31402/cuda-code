// chapter3 片段 6（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 未优化的归约计算，直接使用全局内存
__global__ void globalMemoryReduction(const int *input, int *output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int sharedData[1024]; // 分配共享内存
    if (idx < N) {
        sharedData[threadIdx.x] = input[idx]; // 加载全局内存数据到共享内存
    } else {
        sharedData[threadIdx.x] = 0; // 超出边界的线程初始化为0
    }
    __syncthreads();
    // 分层归约
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // 将每个块的结果写回全局内存
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
// 优化版本：使用共享内存进行归约
__global__ void sharedMemoryReduction(const int *input, int *output, int N) {
    extern __shared__ int sharedData[]; // 动态共享内存
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 加载全局内存数据到共享内存
    if (idx < N) {
        sharedData[threadIdx.x] = input[idx];
    } else {
        sharedData[threadIdx.x] = 0; // 超出边界的线程初始化为0
    }
    __syncthreads();
    // 分层归约
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // 将每个块的结果写回全局内存
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
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
    const int N = 1 << 20; // 数据量 (1M元素)
    const int bytes = N * sizeof(int);
    int *hostInput = new int[N];
    int *hostOutputGlobal = new int[N / 1024];
    int *hostOutputShared = new int[N / 1024];
    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        hostInput[i] = 1; // 数据初始化为1，期望结果为N
    }
    int *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, bytes);
    cudaMalloc(&deviceOutput, bytes / 1024);
    checkCudaError("设备内存分配失败");
    cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    dim3 blockDim(1024);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    auto start = std::chrono::high_resolution_clock::now();
    globalMemoryReduction<<<gridDim, blockDim>>>(deviceInput, deviceOutput, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto globalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostOutputGlobal, deviceOutput, bytes / 1024, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    start = std::chrono::high_resolution_clock::now();
    sharedMemoryReduction<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(deviceInput, deviceOutput, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto sharedDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostOutputShared, deviceOutput, bytes / 1024, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    bool correct = true;
    for (int i = 0; i < gridDim.x; ++i) {
        if (hostOutputGlobal[i] != hostOutputShared[i] || hostOutputGlobal[i] != 1024) {
            correct = false;
            break;
        }
    }
    std::cout << "全局内存版本执行时间: " << globalDuration << " ms" << std::endl;
    std::cout << "共享内存版本执行时间: " << sharedDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutputGlobal;
    delete[] hostOutputShared;
    return 0;
}
