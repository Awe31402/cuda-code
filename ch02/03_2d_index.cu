// chapter2 片段 3：二维线程网格索引映射
#include <cuda_runtime.h>
#include <iostream>
// 核函数：计算二维线程网格的索引并映射到矩阵元素
__global__ void compute2DIndex(int *matrix, int width, int height) {
    // 计算当前线程在二维网格中的全局索引
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    // 确保索引不越界
    if (row < height && col < width) {
        int index = row * width + col; // 将二维坐标映射为一维索引
        matrix[index] = index;        // 将一维索引存入矩阵
    }
}
// 打印二维矩阵
void printMatrix(int *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << "\t";
        }
        std::cout << std::endl;
    }
}
int main() {
    // 矩阵维度
    int width = 8;  // 矩阵的列数
    int height = 8; // 矩阵的行数
    // 分配主机内存
    int *hostMatrix = new int[width * height];
    // 分配设备内存
    int *deviceMatrix;
    cudaMalloc(&deviceMatrix, width * height * sizeof(int));
    // 配置线程网格
    dim3 blockDim(4, 4); // 每个线程块包含4x4个线程
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y); // 网格维度计算
    std::cout << "线程块维度: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    std::cout << "网格维度: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    // 启动核函数
    compute2DIndex<<<gridDim, blockDim>>>(deviceMatrix, width, height);
    cudaDeviceSynchronize();
    // 拷贝结果回主机
    cudaMemcpy(hostMatrix, deviceMatrix, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    // 打印结果
    std::cout << "矩阵索引计算结果:" << std::endl;
    printMatrix(hostMatrix, width, height);
    // 释放内存
    delete[] hostMatrix;
    cudaFree(deviceMatrix);
    return 0;
}
