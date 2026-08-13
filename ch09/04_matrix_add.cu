// chapter9 片段 4（自 src/chapter9.txt 抽出）
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
// 核函数：矩阵加法
__global__ void matrix_add(const float *a, const float *b, float *c, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * cols + idx;
    if (idx < cols && idy < rows) {
        c[index] = a[index] + b[index];
    }
}
// 初始化矩阵
void initialize_matrix(float *matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = value + i % 10;
    }
}
// 分块处理矩阵加法
void block_matrix_addition(int rows, int cols, int block_size) {
    const int matrix_size = rows * cols;
    const int bytes = matrix_size * sizeof(float);
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    initialize_matrix(h_a, rows, cols, 1.0f);
    initialize_matrix(h_b, rows, cols, 2.0f);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    int blocks_per_row = (cols + block_size - 1) / block_size;
    int blocks_per_col = (rows + block_size - 1) / block_size;
    dim3 threads(block_size, block_size);
    dim3 blocks(blocks_per_row, blocks_per_col);
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    auto start = std::chrono::high_resolution_clock::now();
    for (int block_id = 0; block_id < 2; ++block_id) {
        int offset = block_id * (matrix_size / 2);
        cudaMemcpyAsync(d_a + offset, h_a + offset, bytes / 2, cudaMemcpyHostToDevice, streams[block_id]);
        cudaMemcpyAsync(d_b + offset, h_b + offset, bytes / 2, cudaMemcpyHostToDevice, streams[block_id]);
        matrix_add<<<blocks, threads, 0, streams[block_id]>>>(d_a + offset, d_b + offset, d_c + offset, rows / 2, cols);
        cudaMemcpyAsync(h_c + offset, d_c + offset, bytes / 2, cudaMemcpyDeviceToHost, streams[block_id]);
    }
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    auto end = std::chrono::high_resolution_clock::now();
    printf("Execution Time: %.3f ms\n", 
        std::chrono::duration<float, std::milli>(end - start).count());
    bool success = true;
    for (int i = 0; i < matrix_size; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            printf("Error at index %d: %f != %f + %f\n", i, h_c[i], h_a[i], h_b[i]);
            break;
        }
    }
    if (success) {
        printf("Results are correct.\n");
    }
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}
int main() {
    int rows = 1024;
    int cols = 1024;
    int block_size = 16;
    block_matrix_addition(rows, cols, block_size);
    return 0;
}
