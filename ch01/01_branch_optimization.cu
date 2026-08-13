// chapter1 片段 1：分支发散优化 / 表达式合并
// 原文只是片段，这里补成可编译可运行的完整程序。
#include <cuda_runtime.h>
#include <iostream>

// 非优化版本：warp 内偶数/奇数线程走不同分支
__global__ void branchNaive(int *data)
{
    if (threadIdx.x % 2 == 0) {
        data[threadIdx.x] = threadIdx.x * 2;
    } else {
        data[threadIdx.x] = threadIdx.x * 3;
    }
}

// 优化版本：用算术表达式消除分支
__global__ void branchOptimized(int *data)
{
    data[threadIdx.x] = threadIdx.x * (2 + (threadIdx.x % 2));
}

// 非优化：多余的中间变量
__global__ void tempNaive(const float *a, const float *b, const float *c, float *out)
{
    float temp1 = a[threadIdx.x] + b[threadIdx.x];
    float temp2 = temp1 * c[threadIdx.x];
    out[threadIdx.x] = temp2;
}

// 优化：合并成单个表达式（编译器更易生成 FMA）
__global__ void tempOptimized(const float *a, const float *b, const float *c, float *out)
{
    float temp = (a[threadIdx.x] + b[threadIdx.x]) * c[threadIdx.x];
    out[threadIdx.x] = temp;
}

int main()
{
    const int n = 32;
    int h[n];
    float hf[n];

    int *d_data;
    float *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    float init[n];
    for (int i = 0; i < n; ++i) init[i] = static_cast<float>(i);
    cudaMemcpy(d_a, init, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, init, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, init, n * sizeof(float), cudaMemcpyHostToDevice);

    branchNaive<<<1, n>>>(d_data);
    cudaMemcpy(h, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "非优化分支结果: ";
    for (int i = 0; i < 8; ++i) std::cout << h[i] << " ";
    std::cout << std::endl;

    branchOptimized<<<1, n>>>(d_data);
    cudaMemcpy(h, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "优化后分支结果: ";
    for (int i = 0; i < 8; ++i) std::cout << h[i] << " ";
    std::cout << std::endl;

    tempNaive<<<1, n>>>(d_a, d_b, d_c, d_out);
    cudaMemcpy(hf, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "非优化表达式结果: ";
    for (int i = 0; i < 8; ++i) std::cout << hf[i] << " ";
    std::cout << std::endl;

    tempOptimized<<<1, n>>>(d_a, d_b, d_c, d_out);
    cudaMemcpy(hf, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "优化后表达式结果: ";
    for (int i = 0; i < 8; ++i) std::cout << hf[i] << " ";
    std::cout << std::endl;

    cudaFree(d_data);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
    return 0;
}
