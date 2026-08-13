// chapter11 片段 5（自 src/chapter11.txt 抽出）
// mpi_cuda_matrix.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#define N 1024 // 矩阵维度
// CUDA 核函数：矩阵相乘
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) {
        float value = 0;
        for (int k = 0; k < size; k++) {
            value += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = value;
    }
}
// 初始化矩阵
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int blockSize = N / size; // 每个节点处理的矩阵行数
    float *A, *B, *C;
    float *A_block, *C_block;
    // 主节点初始化矩阵
    if (rank == 0) {
        A = (float *)malloc(N * N * sizeof(float));
        B = (float *)malloc(N * N * sizeof(float));
        C = (float *)malloc(N * N * sizeof(float));
        initializeMatrix(A, N);
        initializeMatrix(B, N);
    }
    // 分配子矩阵
    A_block = (float *)malloc(blockSize * N * sizeof(float));
    C_block = (float *)malloc(blockSize * N * sizeof(float));
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, blockSize * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, blockSize * N * sizeof(float));
    // 广播矩阵 B 给所有节点
    if (rank == 0) {
        MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        B = (float *)malloc(N * N * sizeof(float));
        MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    // 分发矩阵 A 的分块到各节点
    MPI_Scatter(A, blockSize * N, MPI_FLOAT, A_block, blockSize * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // 将数据复制到 GPU
    cudaMemcpy(d_A, A_block, blockSize * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    // 配置 CUDA 核函数
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (blockSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // 执行矩阵相乘
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // 将结果从 GPU 复制到主机内存
    cudaMemcpy(C_block, d_C, blockSize * N * sizeof(float), cudaMemcpyDeviceToHost);
    // 收集计算结果到主节点
    MPI_Gather(C_block, blockSize * N, MPI_FLOAT, C, blockSize * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // 主节点打印结果
    if (rank == 0) {
        std::cout << "矩阵 C 的部分数据：" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << C[i] << " ";
        }
        std::cout << std::endl;
        free(A);
        free(B);
        free(C);
    }
    free(A_block);
    free(C_block);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    MPI_Finalize();
    return 0;
}
