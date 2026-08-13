// chapter1 片段 5：最简向量加法（Runtime API）
#include <iostream>
#include <cuda_runtime.h>
__global__ void addKernel(int *c, const int *a, const int *b, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    const int size = 1024;
    // 原文用的是栈上数组，1024 个 int 尚可；这里保持原样
    static int h_a[size], h_b[size], h_c[size];
    int *d_a, *d_b, *d_c;
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
    addKernel<<<size / 256, 256>>>(d_c, d_a, d_b, size);
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        std::cout << h_c[i] << std::endl;
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
