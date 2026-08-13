// chapter12 片段 3（自 src/chapter12.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#define BLOCK_SIZE 16  // 定义线程块大小
// 范德瓦尔斯能量计算核函数
__global__ void computeEnergyKernel(float *positions, float *energies, int numAtoms) {
    // 定义共享内存用于存储当前块的数据
    __shared__ float sharedPosX[BLOCK_SIZE];
    __shared__ float sharedPosY[BLOCK_SIZE];
    __shared__ float sharedPosZ[BLOCK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int globalX = bx * BLOCK_SIZE + tx;
    int globalY = by * BLOCK_SIZE + ty;
    if (globalX >= numAtoms || globalY >= numAtoms) return;
    float energy = 0.0f;
    // 加载当前线程块的分子数据到共享内存
    sharedPosX[tx] = positions[globalX * 3];
    sharedPosY[tx] = positions[globalX * 3 + 1];
    sharedPosZ[tx] = positions[globalX * 3 + 2];
    __syncthreads();
    // 双循环计算范德瓦尔斯能量
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float dx = sharedPosX[i] - positions[globalY * 3];
        float dy = sharedPosY[i] - positions[globalY * 3 + 1];
        float dz = sharedPosZ[i] - positions[globalY * 3 + 2];
        float distSquared = dx * dx + dy * dy + dz * dz;
        float dist = sqrt(distSquared);
        if (dist > 0) energy += 1.0f / dist;  // 简化的范德瓦尔斯能量公式
    }
    // 写入结果
    atomicAdd(&energies[globalX], energy);
    __syncthreads();
}
// 主程序
int main() {
    int numAtoms = 1024;  // 模拟1024个分子
    size_t dataSize = numAtoms * 3 * sizeof(float);
    size_t energySize = numAtoms * sizeof(float);
    // 主机内存分配
    std::vector<float> h_positions(numAtoms * 3, 1.0f);  // 初始化为1.0
    std::vector<float> h_energies(numAtoms, 0.0f);
    // 设备内存分配
    float *d_positions, *d_energies;
    cudaMalloc((void **)&d_positions, dataSize);
    cudaMalloc((void **)&d_energies, energySize);
    // 数据拷贝到设备
    cudaMemcpy(d_positions, h_positions.data(), dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, h_energies.data(), energySize, cudaMemcpyHostToDevice);
    // 定义线程块和网格尺寸
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((numAtoms + BLOCK_SIZE - 1) / BLOCK_SIZE, (numAtoms + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // 调用核函数
    computeEnergyKernel<<<gridSize, blockSize>>>(d_positions, d_energies, numAtoms);
    // 同步设备
    cudaDeviceSynchronize();
    // 拷贝结果回主机
    cudaMemcpy(h_energies.data(), d_energies, energySize, cudaMemcpyDeviceToHost);
    // 打印部分结果
    for (int i = 0; i < 10; i++) {
        std::cout << "Energy[" << i << "] = " << h_energies[i] << std::endl;
    }
    // 释放设备内存
    cudaFree(d_positions);
    cudaFree(d_energies);
    return 0;
}
