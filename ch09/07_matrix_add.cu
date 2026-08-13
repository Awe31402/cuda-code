// chapter9 片段 7（自 src/chapter9.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#define N 1024  // 矩阵维度
#define BLOCK_SIZE 32  // 线程块大小
// 核函数：矩阵加法
__global__ void matrixAdd(const float* A, const float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) {
        int idx = row * size + col;
        C[idx] = A[idx] + B[idx];
    }
}
int main() {
    const int matrixSize = N * N;
    const int matrixBytes = matrixSize * sizeof(float);
    float *h_A, *h_B, *h_C;  // 主机内存
    float *d_A1, *d_B1, *d_C1;  // 设备内存（流1）
    float *d_A2, *d_B2, *d_C2;  // 设备内存（流2）
    cudaStream_t stream1, stream2;  // 创建流
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    // 分配主机内存
    h_A = (float*)malloc(matrixBytes);
    h_B = (float*)malloc(matrixBytes);
    h_C = (float*)malloc(matrixBytes);
    // 初始化矩阵
    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    // 分配设备内存
    cudaMalloc((void**)&d_A1, matrixBytes / 2);
    cudaMalloc((void**)&d_B1, matrixBytes / 2);
    cudaMalloc((void**)&d_C1, matrixBytes / 2);
    cudaMalloc((void**)&d_A2, matrixBytes / 2);
    cudaMalloc((void**)&d_B2, matrixBytes / 2);
    cudaMalloc((void**)&d_C2, matrixBytes / 2);
    // 设置CUDA网格与线程块大小
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N / 2 + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    // 分块传输数据并执行加法
    cudaMemcpyAsync(d_A1, h_A, matrixBytes / 2, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B, matrixBytes / 2, cudaMemcpyHostToDevice, stream1);
    matrixAdd<<<grid, block, 0, stream1>>>(d_A1, d_B1, d_C1, N / 2);
    cudaMemcpyAsync(h_C, d_C1, matrixBytes / 2, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(d_A2, h_A + matrixSize / 2, matrixBytes / 2, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B2, h_B + matrixSize / 2, matrixBytes / 2, cudaMemcpyHostToDevice, stream2);
    matrixAdd<<<grid, block, 0, stream2>>>(d_A2, d_B2, d_C2, N / 2);
    cudaMemcpyAsync(h_C + matrixSize / 2, d_C2, matrixBytes / 2, cudaMemcpyDeviceToHost, stream2);
    // 同步流
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    // 检查结果
    std::cout << "Sample results from C:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";
    // 清理内存与流
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return 0;
}
