// chapter12 片段 7（自 src/chapter12.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
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
                float force = (1.0f / (dist * dist + 1e-6f));  // 示例力计算
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
// 主机代码：多GPU并行实现
int main() {
    const int numParticles = 10000;  // 模拟粒子数量
    const int numGPUs = 2;          // 使用的GPU数量
    const float cutoff = 1.0f;      // 截断距离
    const int blockSize = 256;      // 每个块的线程数
    // 初始化CUDA设备
    cudaSetDevice(0);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < numGPUs) {
        std::cerr << "Error: Insufficient GPUs available." << std::endl;
        return -1;
    }
    // 分配多GPU的粒子数据
    int particlesPerGPU = numParticles / numGPUs;
    std::vector<float *> d_positions(numGPUs), d_forces(numGPUs);
    std::vector<float *> h_positions(numGPUs), h_forces(numGPUs);
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_positions[i], particlesPerGPU * 3 * sizeof(float));
        cudaMalloc(&d_forces[i], particlesPerGPU * 3 * sizeof(float));
        h_positions[i] = (float *)malloc(particlesPerGPU * 3 * sizeof(float));
        h_forces[i] = (float *)malloc(particlesPerGPU * 3 * sizeof(float));
        // 初始化粒子数据
        for (int j = 0; j < particlesPerGPU * 3; j++) {
            h_positions[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
        cudaMemcpy(d_positions[i], h_positions[i], particlesPerGPU * 3 * sizeof(float), cudaMemcpyHostToDevice);
    }
    // 执行多GPU计算
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        int gridSize = (particlesPerGPU + blockSize - 1) / blockSize;
        computeForces<<<gridSize, blockSize>>>(d_positions[i], d_forces[i], particlesPerGPU, cutoff);
    }
    // 同步并拷回结果
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaMemcpy(h_forces[i], d_forces[i], particlesPerGPU * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    // 输出部分结果
    for (int i = 0; i < 10; i++) {
        std::cout << "Particle " << i << ": Force = (" 
                  << h_forces[0][i * 3] << ", " 
                  << h_forces[0][i * 3 + 1] << ", " 
                  << h_forces[0][i * 3 + 2] << ")" << std::endl;
    }
    // 清理资源
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaFree(d_positions[i]);
        cudaFree(d_forces[i]);
        free(h_positions[i]);
        free(h_forces[i]);
    }
    return 0;
}
