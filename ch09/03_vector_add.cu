// chapter9 片段 3（自 src/chapter9.txt 抽出）
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
// 核函数：将两个数组中的元素逐个相加
__global__ void vector_add(const int *a, const int *b, int *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
// 初始化数组
void initialize_array(int *arr, int size, int value) {
    for (int i = 0; i < size; ++i) {
        arr[i] = value + i;
    }
}
// 异步数据传输与核函数并行执行
void async_transfer_and_compute(int *h_a, int *h_b, int *h_c, int size, int bytes) {
    int *d_a, *d_b, *d_c;
    cudaStream_t stream1, stream2;
    // 分配设备内存
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    // 创建流
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    auto start = std::chrono::high_resolution_clock::now();
    // 异步将数据传输到设备
    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream2);
    // 确保数据传输完成后启动核函数
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vector_add<<<blocks, threads, 0, stream1>>>(d_a, d_b, d_c, size);
    // 异步将结果传回主机
    cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream1);
    // 等待所有任务完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    auto end = std::chrono::high_resolution_clock::now();
    printf("Execution Time: %.3f ms\n", 
        std::chrono::duration<float, std::milli>(end - start).count());
    // 销毁流和释放内存
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
int main() {
    const int size = 1024 * 1024; // 1M元素
    const int bytes = size * sizeof(int);
    int *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    initialize_array(h_a, size, 1);
    initialize_array(h_b, size, 2);
    async_transfer_and_compute(h_a, h_b, h_c, size, bytes);
    // 验证结果
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            printf("Error at index %d: %d != %d + %d\n", i, h_c[i], h_a[i], h_b[i]);
            break;
        }
    }
    if (success) {
        printf("Results are correct.\n");
    }
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}
