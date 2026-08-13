// chapter5 片段 7（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
// 核函数：矩阵乘法
__global__ void matrixMul(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            value += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = value;
    }
}
int main() {
    const int N = 256;
    const int bytes = N * N * sizeof(float);
    float *hostA = new float[N * N];
    float *hostB = new float[N * N];
    float *hostC = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        hostA[i] = static_cast<float>(i % 100);
        hostB[i] = static_cast<float>((i + 1) % 100);
    }
    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);
    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    matrixMul<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, N);
    cudaDeviceSynchronize();
    cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "]: " << hostC[i] << std::endl;
    }
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
