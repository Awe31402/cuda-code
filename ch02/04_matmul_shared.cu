// chapter2 片段 4：基于共享内存分块的矩阵乘法
// 注意：原文会把 3 个 256x256 矩阵全部打印出来（约 20 万个数字），
// 运行时建议重定向输出：./04_matmul_shared > out.txt
#include <cuda_runtime.h>
#include <iostream>
// 矩阵维度
const int MATRIX_SIZE = 256; // 假设为方阵
// 核函数：基于共享内存的矩阵乘法
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int N) {
    // 共享内存分配
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];
    // 当前线程的行和列索引
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float result = 0.0f; // 用于存储C[row][col]的值
    // 分块加载A和B的子矩阵并进行计算
    for (int tile = 0; tile < (N + 15) / 16; ++tile) {
        // 加载A的子块到共享内存
        if (row < N && tile * 16 + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tile * 16 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // 加载B的子块到共享内存
        if (tile * 16 + threadIdx.y < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads(); // 确保所有线程加载完成
        // 计算C[row][col]
        for (int k = 0; k < 16; ++k) {
            result += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads(); // 确保所有线程完成计算
    }
    // 将结果写入C矩阵
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}
// 初始化矩阵
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<float>(rand() % 10); // 随机数在0-9之间
    }
}
// 打印矩阵
void printMatrix(const float *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << "\t";
        }
        std::cout << std::endl;
    }
}
int main() {
    int N = MATRIX_SIZE;
    // 分配主机内存
    float *hostA = new float[N * N];
    float *hostB = new float[N * N];
    float *hostC = new float[N * N];
    // 初始化矩阵A和B
    initializeMatrix(hostA, N);
    initializeMatrix(hostB, N);
    // 分配设备内存
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, N * N * sizeof(float));
    cudaMalloc(&deviceB, N * N * sizeof(float));
    cudaMalloc(&deviceC, N * N * sizeof(float));
    // 拷贝数据到设备
    cudaMemcpy(deviceA, hostA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N * N * sizeof(float), cudaMemcpyHostToDevice);
    // 配置线程网格
    dim3 blockDim(16, 16); // 每个线程块16x16线程
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    // 启动核函数
    matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    // 拷贝结果回主机
    cudaMemcpy(hostC, deviceC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    // 打印矩阵
    std::cout << "矩阵A:" << std::endl;
    printMatrix(hostA, N);
    std::cout << "矩阵B:" << std::endl;
    printMatrix(hostB, N);
    std::cout << "矩阵C (结果):" << std::endl;
    printMatrix(hostC, N);
    // 释放内存
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
