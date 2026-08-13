// chapter5 片段 9（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 宏函数：通用错误检测
#define CUDA_CHECK(call)                                                               \
    {                                                                                  \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << cudaGetErrorString(err) << std::endl;                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    }
// 核函数：矩阵加法
__global__ void matrixAdd(const float *a, const float *b, float *c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
        // 调试输出：每个线程的执行路径
        printf("Thread (%d, %d) calculated c[%d] = %f\n", row, col, idx, c[idx]);
    }
}
int main() {
    const int rows = 4, cols = 4; // 示例矩阵大小
    const int size = rows * cols * sizeof(float);
    // 主机内存分配
    float hostA[rows * cols], hostB[rows * cols], hostC[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    // 设备内存分配
    float *deviceA, *deviceB, *deviceC;
    CUDA_CHECK(cudaMalloc((void **)&deviceA, size));
    CUDA_CHECK(cudaMalloc((void **)&deviceB, size));
    CUDA_CHECK(cudaMalloc((void **)&deviceC, size));
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice));
    // 设置线程块和网格大小
    dim3 blockSize(2, 2);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    // 启动核函数
    matrixAdd<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, rows, cols);
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
    // 数据传输回主机
    CUDA_CHECK(cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost));
    // 打印结果
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << hostC[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    // 释放内存
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    return 0;
}
