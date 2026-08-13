// chapter8 片段 4（自 src/chapter8.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#define CUDA_CHECK(call)                                                                 \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }
// 核函数：线程块内使用共享内存进行归约
__global__ void reduceSumWithSync(int *input, int *output, int N) {
    // 分配共享内存
    __shared__ int sharedData[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 将全局内存数据加载到共享内存
    if (tid < N) {
        sharedData[threadIdx.x] = input[tid];
    } else {
        sharedData[threadIdx.x] = 0; // 超出数据范围时初始化为0
    }
    __syncthreads(); // 确保共享内存加载完成
    // 使用二分归约算法在共享内存中计算
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        }
        __syncthreads(); // 确保每轮计算完成后再进行下一轮
    }
    // 将每个线程块的归约结果写入全局内存
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
int main() {
    const int N = 1024; // 数据大小
    const int bytes = N * sizeof(int);
    // 主机内存分配
    int *hostInput = new int[N];
    int *hostOutput = new int[(N + 255) / 256];
    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        hostInput[i] = 1; // 所有元素初始化为1
    }
    // 设备内存分配
    int *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, sizeof(int) * (N + 255) / 256));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    // 配置线程块和网格
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // 执行核函数
    reduceSumWithSync<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 将结果传回主机
    CUDA_CHECK(cudaMemcpy(hostOutput, deviceOutput, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost));
    // 最终归约结果
    int finalSum = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        finalSum += hostOutput[i];
    }
    // 打印结果
    std::cout << "Final Sum: " << finalSum << std::endl;
    // 清理内存
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    delete[] hostInput;
    delete[] hostOutput;
    return 0;
}
