// chapter9 片段 1（自 src/chapter9.txt 抽出）
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void kernel_add(int *a, int *b, int *c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
void initialize_array(int *arr, int size, int value) {
    for (int i = 0; i < size; ++i) {
        arr[i] = value + i;
    }
}
int main() {
    const int size = 1024;
    const int bytes = size * sizeof(int);
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    initialize_array(h_a, size, 1);
    initialize_array(h_b, size, 10);
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream2);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel_add<<<blocks, threads, 0, stream1>>>(d_a, d_b, d_c, size / 2);
    kernel_add<<<blocks, threads, 0, stream2>>>(d_a + size / 2, d_b + size / 2, d_c + size / 2, size / 2);
    cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_c + size / 2, d_c + size / 2, bytes / 2, cudaMemcpyDeviceToHost, stream2);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    printf("Result: ");
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_c[i]);
    }
    printf("\n");
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
