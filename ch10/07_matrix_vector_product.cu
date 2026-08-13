// chapter10 片段 7（自 src/chapter10.txt 抽出）
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#define N 1024 // 维度大小
#define MAX_ITER 1000
#define EPSILON 1e-6
__global__ void matrixVectorProduct(const float* A, const float* x, float* Ax, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            sum += A[row * n + col] * x[col];
        }
        Ax[row] = sum;
    }
}
__global__ void vectorUpdate(float* x, const float* p, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += alpha * p[idx];
    }
}
int main() {
    // 初始化变量
    float *h_A, *h_b, *h_x;
    float *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap;
    float alpha, beta, r_dot_r, r_dot_r_new;
    // 主机内存分配
    h_A = new float[N * N];
    h_b = new float[N];
    h_x = new float[N];
    // 随机初始化A和b
    for (int i = 0; i < N; ++i) {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    // 设备内存分配
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_r, N * sizeof(float));
    cudaMalloc(&d_p, N * sizeof(float));
    cudaMalloc(&d_Ap, N * sizeof(float));
    // 数据拷贝到设备
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, N * sizeof(float)); // 初始化x为0
    // cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    // 初始化r和p为b
    cudaMemcpy(d_r, d_b, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_b, N * sizeof(float), cudaMemcpyDeviceToDevice);
    // 计算初始r_dot_r
    cublasSdot(handle, N, d_r, 1, d_r, 1, &r_dot_r);
    int iter = 0;
    while (r_dot_r > EPSILON * EPSILON && iter < MAX_ITER) {
        // Ap = A * p
        matrixVectorProduct<<<(N + 255) / 256, 256>>>(d_A, d_p, d_Ap, N);
        // alpha = r_dot_r / (p^T * Ap)
        float p_dot_Ap;
        cublasSdot(handle, N, d_p, 1, d_Ap, 1, &p_dot_Ap);
        alpha = r_dot_r / p_dot_Ap;
        // x = x + alpha * p
        vectorUpdate<<<(N + 255) / 256, 256>>>(d_x, d_p, alpha, N);
        // r = r - alpha * Ap
        vectorUpdate<<<(N + 255) / 256, 256>>>(d_r, d_Ap, -alpha, N);
        // r_dot_r_new = r^T * r
        cublasSdot(handle, N, d_r, 1, d_r, 1, &r_dot_r_new);
        // beta = r_dot_r_new / r_dot_r
        beta = r_dot_r_new / r_dot_r;
        // p = r + beta * p
        cublasSscal(handle, N, &beta, d_p, 1);
        cublasSaxpy(handle, N, &alpha, d_r, 1, d_p, 1);
        r_dot_r = r_dot_r_new;
        iter++;
    }
    // 拷贝结果回主机
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Conjugate Gradient completed in " << iter << " iterations." << std::endl;
    std::cout << "Solution vector (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_x[i] << " ";
    }
    std::cout << std::endl;
    // 释放资源
    delete[] h_A;
    delete[] h_b;
    delete[] h_x;
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cublasDestroy(handle);
    return 0;
}
