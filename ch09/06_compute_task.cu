// chapter9 片段 6（自 src/chapter9.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#define DATA_SIZE 1024
#define BLOCK_SIZE 256
// 核函数模拟不同任务
__global__ void computeTask(float* data, int offset, float factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < DATA_SIZE) {
        data[idx] += (offset + idx) * factor;
    }
}
int main() {
    float *d_data1, *d_data2, *d_data3;
    float h_data1[DATA_SIZE], h_data2[DATA_SIZE], h_data3[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; i++) {
        h_data1[i] = 0.0f;
        h_data2[i] = 0.0f;
        h_data3[i] = 0.0f;
    }
    cudaMalloc((void**)&d_data1, DATA_SIZE * sizeof(float));
    cudaMalloc((void**)&d_data2, DATA_SIZE * sizeof(float));
    cudaMalloc((void**)&d_data3, DATA_SIZE * sizeof(float));
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaMemcpyAsync(d_data1, h_data1, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_data2, h_data2, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_data3, h_data3, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream3);
    computeTask<<<(DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>>(d_data1, 1, 1.5f);
    computeTask<<<(DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream2>>>(d_data2, 2, 2.0f);
    computeTask<<<(DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream3>>>(d_data3, 3, 2.5f);
    cudaMemcpyAsync(h_data1, d_data1, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_data2, d_data2, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(h_data3, d_data3, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream3);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    std::cout << "Stream1 results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data1[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Stream2 results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data2[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Stream3 results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data3[i] << " ";
    }
    std::cout << "\n";
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
    return 0;
}
