// chapter12 片段 2（自 src/chapter12.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#define N 1024  // 总分子数
#define BLOCK_SIZE 256  // 每个线程块的线程数
__global__ void calculateForces(float* positions, float* forces, int numParticles) {
    __shared__ float sharedPos[BLOCK_SIZE][3];  // 共享内存存储分子位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx >= numParticles) return;
    // 加载当前分子位置到共享内存
    sharedPos[tid][0] = positions[idx * 3 + 0];
    sharedPos[tid][1] = positions[idx * 3 + 1];
    sharedPos[tid][2] = positions[idx * 3 + 2];
    __syncthreads();
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    for (int j = 0; j < blockDim.x; j++) {
        if (j + blockIdx.x * blockDim.x >= numParticles) continue;
        float dx = sharedPos[tid][0] - sharedPos[j][0];
        float dy = sharedPos[tid][1] - sharedPos[j][1];
        float dz = sharedPos[tid][2] - sharedPos[j][2];
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0.0001f) {
            float r6 = r2 * r2 * r2;
            float force = 24.0f * (2.0f / r6 - 1.0f) / (r6 * r2);
            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
        }
    }
    forces[idx * 3 + 0] = fx;
    forces[idx * 3 + 1] = fy;
    forces[idx * 3 + 2] = fz;
}
int main() {
    float* h_positions = new float[N * 3];
    float* h_forces = new float[N * 3];
    float* d_positions;
    float* d_forces;
    for (int i = 0; i < N * 3; i++) {
        h_positions[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    cudaMalloc(&d_positions, N * 3 * sizeof(float));
    cudaMalloc(&d_forces, N * 3 * sizeof(float));
    cudaMemcpy(d_positions, h_positions, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    calculateForces<<<gridSize, blockSize>>>(d_positions, d_forces, N);
    cudaMemcpy(h_forces, d_forces, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Forces on the first few molecules:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Molecule " << i << ": (" << h_forces[i * 3 + 0] << ", "
                  << h_forces[i * 3 + 1] << ", " << h_forces[i * 3 + 2] << ")" << std::endl;
    }
    delete[] h_positions;
    delete[] h_forces;
    cudaFree(d_positions);
    cudaFree(d_forces);
    return 0;
}
