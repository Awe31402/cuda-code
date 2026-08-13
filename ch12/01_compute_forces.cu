// chapter12 片段 1（自 src/chapter12.txt 抽出）
// 引入必要的CUDA和标准库头文件
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
// 定义常量
#define NUM_PARTICLES 1024  // 分子数量
#define BLOCK_SIZE 256     // 每个线程块的线程数
#define EPSILON 1e-8       // 避免除以零的小常量
// GPU核函数：计算分子间作用力
__global__ void computeForces(float3* positions, float3* forces, int numParticles) {
    // 获取线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    float3 myPosition = positions[idx];
    float3 force = {0.0f, 0.0f, 0.0f};
    // 计算与其他分子的作用力
    for (int j = 0; j < numParticles; ++j) {
        if (j == idx) continue;
        float3 otherPosition = positions[j];
        float dx = myPosition.x - otherPosition.x;
        float dy = myPosition.y - otherPosition.y;
        float dz = myPosition.z - otherPosition.z;
        float distSqr = dx * dx + dy * dy + dz * dz + EPSILON;
        float distInv = rsqrtf(distSqr);
        float forceMagnitude = distInv * distInv;  // 假设简单的1/r^2力模型
        force.x += forceMagnitude * dx * distInv;
        force.y += forceMagnitude * dy * distInv;
        force.z += forceMagnitude * dz * distInv;
    }
    // 保存计算结果
    forces[idx] = force;
}
// 主函数
int main() {
    // 定义主机内存变量
    float3* h_positions = new float3[NUM_PARTICLES];
    float3* h_forces = new float3[NUM_PARTICLES];
    // 初始化分子位置
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_positions[i].x = static_cast<float>(rand()) / RAND_MAX;
        h_positions[i].y = static_cast<float>(rand()) / RAND_MAX;
        h_positions[i].z = static_cast<float>(rand()) / RAND_MAX;
        h_forces[i] = {0.0f, 0.0f, 0.0f};
    }
    // 定义设备内存变量
    float3* d_positions;
    float3* d_forces;
    // 分配设备内存
    cudaMalloc((void**)&d_positions, NUM_PARTICLES * sizeof(float3));
    cudaMalloc((void**)&d_forces, NUM_PARTICLES * sizeof(float3));
    // 将主机数据拷贝到设备
    cudaMemcpy(d_positions, h_positions, NUM_PARTICLES * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemset(d_forces, 0, NUM_PARTICLES * sizeof(float3));
    // 定义线程块与网格
    int numBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // 启动核函数
    computeForces<<<numBlocks, BLOCK_SIZE>>>(d_positions, d_forces, NUM_PARTICLES);
    // 同步设备，确保核函数完成
    cudaDeviceSynchronize();
    // 将结果从设备拷贝回主机
    cudaMemcpy(h_forces, d_forces, NUM_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost);
    // 输出部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << "Particle " << i << " Force: "
                  << h_forces[i].x << ", "
                  << h_forces[i].y << ", "
                  << h_forces[i].z << std::endl;
    }
    // 释放设备内存
    cudaFree(d_positions);
    cudaFree(d_forces);
    // 释放主机内存
    delete[] h_positions;
    delete[] h_forces;
    return 0;
}
