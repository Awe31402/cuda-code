// chapter2 片段 8：动态并行递归计算每行平方和（原文代码，需 -rdc=true）
// 已知问题：同 07，设备端 cudaDeviceSynchronize() 在 CUDA 12 已移除。
// 可运行版见 08_dynamic_parallel_rowsum_fixed.cu
#include <cuda_runtime.h>
#include <iostream>
// 动态并行核函数：递归计算二维矩阵每行的平方和
__global__ void recursiveRowSum(int *matrix, int rows, int cols, int *result) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // 如果当前线程超出行范围，直接返回
    if (rowIdx >= rows) return;
    // 初始化当前行的平方和
    int sum = 0;
    for (int col = 0; col < cols; ++col) {
        int idx = rowIdx * cols + col; // 计算矩阵元素的线性索引
        sum += matrix[idx] * matrix[idx];
    }
    // 将结果写入中间数组
    result[rowIdx] = sum;
    // 如果是第一层递归，启动动态网格进行下一步递归
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int newCols = cols / 2; // 每层递归减少列数
        if (newCols > 0) {
            recursiveRowSum<<<1, rows>>>(matrix, rows, newCols, result);
            cudaDeviceSynchronize();
        }
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
    const int rows = 4, cols = 8; // 矩阵大小
    int hostMatrix[rows][cols] = { // 初始化二维矩阵
        {1, 2, 3, 4, 5, 6, 7, 8},
        {2, 4, 6, 8, 10, 12, 14, 16},
        {3, 6, 9, 12, 15, 18, 21, 24},
        {4, 8, 12, 16, 20, 24, 28, 32}
    };
    int hostResult[rows]; // 存储最终结果
    // 分配设备内存
    int *deviceMatrix, *deviceResult;
    cudaMalloc(&deviceMatrix, rows * cols * sizeof(int));
    cudaMalloc(&deviceResult, rows * sizeof(int));
    checkCudaError("设备内存分配失败");
    // 拷贝矩阵到设备
    cudaMemcpy(deviceMatrix, hostMatrix, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("主机到设备数据传输失败");
    // 启动核函数
    recursiveRowSum<<<1, rows>>>(deviceMatrix, rows, cols, deviceResult);
    cudaDeviceSynchronize();
    checkCudaError("核函数执行失败");
    // 拷贝结果回主机
    cudaMemcpy(hostResult, deviceResult, rows * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError("设备到主机数据传输失败");
    // 输出结果
    std::cout << "每行的平方和结果:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        std::cout << "第 " << i + 1 << " 行: " << hostResult[i] << std::endl;
    }
    // 释放内存
    cudaFree(deviceMatrix);
    cudaFree(deviceResult);
    return 0;
}
