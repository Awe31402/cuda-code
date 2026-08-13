// chapter2 片段 8（CUDA 12 可运行版）：动态并行递归计算每行平方和
// 与原文的差异：
//   1. 删掉设备端 cudaDeviceSynchronize()（CUDA 12 已移除）。
//   2. 在启动子网格前加 __syncthreads()，避免"子网格写 result"与
//      "父网格其余线程写 result"竞争（原文这里本身就是竞态）。
// 保留的原文行为：每层递归把 cols 减半，但线性索引仍按当前 cols 计算，
// 所以最终 result[i] 是 cols==1 那一层的结果，即 matrix[i] 的平方，
// 而不是整行的平方和。这是原文算法本身的问题，此处如实保留。
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
    __syncthreads(); // 本层所有线程写完后再启动下一层
    // 如果是第一层递归，启动动态网格进行下一步递归
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int newCols = cols / 2; // 每层递归减少列数
        if (newCols > 0) {
            recursiveRowSum<<<1, rows>>>(matrix, rows, newCols, result);
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
    // 参考：真正的整行平方和（用于对比）
    int reference[rows] = {0, 0, 0, 0};
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            reference[i] += hostMatrix[i][j] * hostMatrix[i][j];
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
    std::cout << "递归最后一层(cols=1)的结果:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        std::cout << "第 " << i + 1 << " 行: " << hostResult[i]
                  << "  (整行平方和应为 " << reference[i] << ")" << std::endl;
    }
    // 释放内存
    cudaFree(deviceMatrix);
    cudaFree(deviceResult);
    return 0;
}
