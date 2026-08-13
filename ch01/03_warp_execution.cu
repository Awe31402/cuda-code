// chapter1 片段 3：Warp 分支发散模拟 + 寄存器分配测试
#include <cuda_runtime.h>
#include <iostream>
// 检查CUDA错误宏
#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "      \
                      << cudaGetErrorString(error) << std::endl;               \
            exit(1);                                                           \
        }                                                                      \
    }
// 核函数模拟Warp分组与分支发散
__global__ void warpExecutionSimulation(int *data, int n)
{
    // 当前线程的索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 模拟Warp的分支行为
    if (idx < n)
    {
        if (idx % 2 == 0)
        {
            data[idx] = idx * 2; // 偶数线程
        }
        else
        {
            data[idx] = idx * 3; // 奇数线程
        }
    }
}
// 核函数演示寄存器溢出和局部内存行为
__global__ void registerAllocationTest(int *data, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        // 模拟寄存器高占用导致的溢出
        int temp1 = idx + 1;
        int temp2 = temp1 * 2;
        int temp3 = temp2 + idx;
        // 使用计算结果更新数据
        data[idx] = temp3;
    }
}
// 结果打印函数
void printResults(const int *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        std::cout << "Index " << i << ": Value " << data[i] << std::endl;
    }
}
// 主函数
int main()
{
    // 数据大小
    const int n = 32;
    const int size = n * sizeof(int);
    // 主机和设备内存
    int *h_data = (int *)malloc(size);
    int *d_data;
    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void **)&d_data, size));
    // 初始化主机数据
    for (int i = 0; i < n; i++)
    {
        h_data[i] = 0;
    }
    // 传输数据到设备
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 blockSize(32);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    // 执行核函数
    std::cout << "运行Warp分支模拟核函数..." << std::endl;
    warpExecutionSimulation<<<gridSize, blockSize>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 传输结果回主机
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    // 打印Warp分支模拟结果
    std::cout << "Warp分支模拟结果:" << std::endl;
    printResults(h_data, n);
    // 再次初始化数据
    for (int i = 0; i < n; i++)
    {
        h_data[i] = 0;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    // 执行寄存器溢出测试核函数
    std::cout << "运行寄存器分配测试核函数..." << std::endl;
    registerAllocationTest<<<gridSize, blockSize>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 传输结果回主机
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    // 打印寄存器分配测试结果
    std::cout << "寄存器分配测试结果:" << std::endl;
    printResults(h_data, n);
    // 清理内存
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
