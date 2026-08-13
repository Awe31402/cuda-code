// chapter12 片段 8（自 src/chapter12.txt 抽出）
#include <cuda_runtime.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
// GPU内核函数：计算分子间作用力
__global__ void computeForces(float *positions, float *forces, int numParticles, float cutoff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        float xi = positions[idx * 3];
        float yi = positions[idx * 3 + 1];
        float zi = positions[idx * 3 + 2];
        for (int j = 0; j < numParticles; j++) {
            if (j == idx) continue;
            float xj = positions[j * 3];
            float yj = positions[j * 3 + 1];
            float zj = positions[j * 3 + 2];
            float dx = xj - xi;
            float dy = yj - yi;
            float dz = zj - zi;
            float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < cutoff * cutoff) {
                float dist = sqrtf(distSq);
                float force = (1.0f / (dist * dist + 1e-6f));
                fx += force * dx / dist;
                fy += force * dy / dist;
                fz += force * dz / dist;
            }
        }
        forces[idx * 3] = fx;
        forces[idx * 3 + 1] = fy;
        forces[idx * 3 + 2] = fz;
    }
}
// 主机代码：多GPU协同计算与性能验证
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    const int numParticles = 10000;  // 模拟粒子总数
    const int blockSize = 256;       // 每个块的线程数
    const float cutoff = 1.0f;       // 截断距离
    int particlesPerGPU = numParticles / worldSize;
    float *d_positions, *d_forces;
    float *h_positions = (float *)malloc(particlesPerGPU * 3 * sizeof(float));
    float *h_forces = (float *)malloc(particlesPerGPU * 3 * sizeof(float));
    // 初始化粒子数据
    for (int i = 0; i < particlesPerGPU * 3; i++) {
        h_positions[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    // 设置GPU并分配内存
    cudaSetDevice(worldRank);
    cudaMalloc(&d_positions, particlesPerGPU * 3 * sizeof(float));
    cudaMalloc(&d_forces, particlesPerGPU * 3 * sizeof(float));
    cudaMemcpy(d_positions, h_positions, particlesPerGPU * 3 * sizeof(float), cudaMemcpyHostToDevice);
    // 计算分子间作用力
    int gridSize = (particlesPerGPU + blockSize - 1) / blockSize;
    computeForces<<<gridSize, blockSize>>>(d_positions, d_forces, particlesPerGPU, cutoff);
    // 同步数据到主机并通过MPI通信合并结果
    cudaMemcpy(h_forces, d_forces, particlesPerGPU * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    float *globalForces = nullptr;
    if (worldRank == 0) {
        globalForces = (float *)malloc(numParticles * 3 * sizeof(float));
    }
    MPI_Gather(h_forces, particlesPerGPU * 3, MPI_FLOAT, globalForces, particlesPerGPU * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // 输出结果与验证
    if (worldRank == 0) {
        for (int i = 0; i < 10; i++) {
            std::cout << "Particle " << i << ": Force = (" 
                      << globalForces[i * 3] << ", " 
                      << globalForces[i * 3 + 1] << ", " 
                      << globalForces[i * 3 + 2] << ")" << std::endl;
        }
        free(globalForces);
    }
    // 清理资源
    cudaFree(d_positions);
    cudaFree(d_forces);
    free(h_positions);
    free(h_forces);
    MPI_Finalize();
    return 0;
}
