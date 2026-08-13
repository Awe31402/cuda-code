// chapter11 片段 6（自 src/chapter11.txt 抽出）
#include <mpi.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
// 矩阵维度
#define N 1024
// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
// 初始化矩阵
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}
// CUDA 核函数：转置矩阵
__global__ void transposeKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}
// 转置矩阵
void transposeMatrix(float *d_input, float *d_output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
}
// 主程序
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int blockSize = N / size; // 每个节点的分块行数
    float *A, *A_block, *A_block_T;
    float *d_A_block, *d_A_block_T;
    // 主节点初始化矩阵
    if (rank == 0) {
        A = (float *)malloc(N * N * sizeof(float));
        initializeMatrix(A, N, N);
    }
    // 分配分块
    A_block = (float *)malloc(blockSize * N * sizeof(float));
    A_block_T = (float *)malloc(blockSize * N * sizeof(float));
    CUDA_CHECK(cudaMalloc((void **)&d_A_block, blockSize * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_A_block_T, blockSize * N * sizeof(float)));
    // 分发矩阵 A 的分块
    MPI_Scatter(A, blockSize * N, MPI_FLOAT, A_block, blockSize * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // 将分块数据复制到 GPU
    CUDA_CHECK(cudaMemcpy(d_A_block, A_block, blockSize * N * sizeof(float), cudaMemcpyHostToDevice));
    // 执行矩阵转置
    transposeMatrix(d_A_block, d_A_block_T, blockSize, N);
    // 将结果从 GPU 复制到主机内存
    CUDA_CHECK(cudaMemcpy(A_block_T, d_A_block_T, blockSize * N * sizeof(float), cudaMemcpyDeviceToHost));
    // 收集转置后的结果
    MPI_Gather(A_block_T, blockSize * N, MPI_FLOAT, A, blockSize * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // 主节点打印部分结果
    if (rank == 0) {
        std::cout << "广义逆矩阵部分数据：" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << A[i] << " ";
        }
        std::cout << std::endl;
        free(A);
    }
    free(A_block);
    free(A_block_T);
    CUDA_CHECK(cudaFree(d_A_block));
    CUDA_CHECK(cudaFree(d_A_block_T));
    MPI_Finalize();
    return 0;
}
