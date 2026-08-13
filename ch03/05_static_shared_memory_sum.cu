// chapter3 片段 5（自 src/chapter3.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
// 核函数：使用静态共享内存
__global__ void staticSharedMemorySum(const int *input, int *output, int N) {
    __shared__ int sharedData[1024]; // 静态分配共享内存
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        sharedData[tid] = input[idx]; // 加载全局内存到共享内存
    } else {
        sharedData[tid] = 0;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0]; // 归约结果写回全局内存
    }
}
// 核函数：使用动态共享内存
__global__ void dynamicSharedMemorySum(const int *input, int *output, int N) {
    extern __shared__ int sharedData[]; // 动态分配共享内存
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        sharedData[tid] = input[idx]; // 加载全局内存到共享内存
    } else {
        sharedData[tid] = 0;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0]; // 归约结果写回全局内存
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
    int *hostOutputStatic = new int[N / 1024];
    int *hostOutputDynamic = new int[N / 1024];
    for (int i = 0; i < N; ++i) {
        hostInput[i] = 1; // 初始化为1，归约结果应为N
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
    staticSharedMemorySum<<<gridDim, blockDim>>>(deviceInput, deviceOutput, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto staticDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostOutputStatic, deviceOutput, bytes / 1024, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    start = std::chrono::high_resolution_clock::now();
    dynamicSharedMemorySum<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(deviceInput, deviceOutput, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto dynamicDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaMemcpy(hostOutputDynamic, deviceOutput, bytes / 1024, cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    bool correct = true;
    for (int i = 0; i < gridDim.x; ++i) {
        if (hostOutputStatic[i] != hostOutputDynamic[i] || hostOutputStatic[i] != 1024) {
            correct = false;
            break;
        }
    }
    std::cout << "静态共享内存版本执行时间: " << staticDuration << " ms" << std::endl;
    std::cout << "动态共享内存版本执行时间: " << dynamicDuration << " ms" << std::endl;
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    delete[] hostOutputStatic;
    delete[] hostOutputDynamic;
    return 0;
}
