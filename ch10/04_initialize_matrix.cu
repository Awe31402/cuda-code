// chapter10 片段 4（自 src/chapter10.txt 抽出）
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(EXIT_FAILURE); \
    }
#define CHECK_CUBLAS(call) \
    if ((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << std::endl; \
        exit(EXIT_FAILURE); \
    }
void initializeMatrix(float* matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = value;
    }
}
void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
int main() {
    const int M = 4, N = 3, K = 5;
    const float alpha = 2.0f, beta = 0.5f;
    float h_A[M * K], h_B[K * N], h_C[M * N];
    initializeMatrix(h_A, M, K, 1.0f);
    initializeMatrix(h_B, K, N, 2.0f);
    initializeMatrix(h_C, M, N, 1.0f);
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_A, M,
        d_B, K,
        &beta,
        d_C, M
    ));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, M, K);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, K, N);
    std::cout << "Result Matrix C (2.0 * A * B + 0.5 * C):" << std::endl;
    printMatrix(h_C, M, N);
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
