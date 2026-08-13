// chapter12 片段 4（自 src/chapter12.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#define BLOCK_SIZE 16  // 定义线程块大小
// 力矩计算核函数
__global__ void computeTorqueKernel(float *positions, float *forces, float *torques, int numAtoms) {
    // 定义共享内存用于存储当前块的位置和力
    __shared__ float sharedPosX[BLOCK_SIZE];
    __shared__ float sharedPosY[BLOCK_SIZE];
    __shared__ float sharedPosZ[BLOCK_SIZE];
    __shared__ float sharedForceX[BLOCK_SIZE];
    __shared__ float sharedForceY[BLOCK_SIZE];
    __shared__ float sharedForceZ[BLOCK_SIZE];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = bx * BLOCK_SIZE + tx;
    if (idx >= numAtoms) return;
    // 加载数据到共享内存
    sharedPosX[tx] = positions[idx * 3];
    sharedPosY[tx] = positions[idx * 3 + 1];
    sharedPosZ[tx] = positions[idx * 3 + 2];
    sharedForceX[tx] = forces[idx * 3];
    sharedForceY[tx] = forces[idx * 3 + 1];
    sharedForceZ[tx] = forces[idx * 3 + 2];
    __syncthreads();
    // 计算力矩
    float torqueX = 0.0f, torqueY = 0.0f, torqueZ = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        float dx = sharedPosY[tx] * sharedForceZ[i] - sharedPosZ[tx] * sharedForceY[i];
        float dy = sharedPosZ[tx] * sharedForceX[i] - sharedPosX[tx] * sharedForceZ[i];
        float dz = sharedPosX[tx] * sharedForceY[i] - sharedPosY[tx] * sharedForceX[i];
        torqueX += dx;
        torqueY += dy;
        torqueZ += dz;
    }
    // 写回全局内存
    torques[idx * 3] = torqueX;
    torques[idx * 3 + 1] = torqueY;
    torques[idx * 3 + 2] = torqueZ;
}
// 主程序
int main() {
    int numAtoms = 1024;  // 模拟1024个分子
    size_t dataSize = numAtoms * 3 * sizeof(float);
    // 主机内存分配
    std::vector<float> h_positions(numAtoms * 3, 1.0f);  // 初始化位置为1.0
    std::vector<float> h_forces(numAtoms * 3, 0.5f);     // 初始化力为0.5
    std::vector<float> h_torques(numAtoms * 3, 0.0f);    // 初始化力矩为0.0
    // 设备内存分配
    float *d_positions, *d_forces, *d_torques;
    cudaMalloc((void **)&d_positions, dataSize);
    cudaMalloc((void **)&d_forces, dataSize);
    cudaMalloc((void **)&d_torques, dataSize);
    // 数据拷贝到设备
    cudaMemcpy(d_positions, h_positions.data(), dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_forces, h_forces.data(), dataSize, cudaMemcpyHostToDevice);
    // 定义线程块和网格尺寸
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((numAtoms + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // 调用核函数
    computeTorqueKernel<<<gridSize, blockSize>>>(d_positions, d_forces, d_torques, numAtoms);
    // 同步设备
    cudaDeviceSynchronize();
    // 拷贝结果回主机
    cudaMemcpy(h_torques.data(), d_torques, dataSize, cudaMemcpyDeviceToHost);
    // 打印部分结果
    for (int i = 0; i < 10; i++) {
        std::cout << "Torque[" << i << "] = (" << h_torques[i * 3] << ", " << h_torques[i * 3 + 1] << ", " << h_torques[i * 3 + 2] << ")" << std::endl;
    }
    // 释放设备内存
    cudaFree(d_positions);
    cudaFree(d_forces);
    cudaFree(d_torques);
    return 0;
}
