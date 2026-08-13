// chapter12 片段 6（自 src/chapter12.txt 抽出）
#include <iostream>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <vector>
// 定义常量
const int numParticles = 1000;
const int blockSize = 256;
// CUDA核函数，用于计算作用力
__global__ void computeForces(const double* positions, double* forces, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        double force = 0.0;
        for (int j = 0; j < numParticles; ++j) {
            if (j != idx) {
                double dx = positions[idx * 3] - positions[j * 3];
                double dy = positions[idx * 3 + 1] - positions[j * 3 + 1];
                double dz = positions[idx * 3 + 2] - positions[j * 3 + 2];
                double dist = sqrt(dx * dx + dy * dy + dz * dz);
                if (dist > 0.1) {
                    force += 24 * ((2 / pow(dist, 13)) - (1 / pow(dist, 7)));
                }
            }
        }
        forces[idx] = force;
    }
}
int main() {
    // 初始化数据
    std::vector<double> h_positions(numParticles * 3);
    std::vector<double> h_forces(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        h_positions[i * 3] = rand() / double(RAND_MAX);
        h_positions[i * 3 + 1] = rand() / double(RAND_MAX);
        h_positions[i * 3 + 2] = rand() / double(RAND_MAX);
    }
    // 分配设备内存
    double* d_positions;
    double* d_forces;
    cudaMalloc(&d_positions, numParticles * 3 * sizeof(double));
    cudaMalloc(&d_forces, numParticles * sizeof(double));
    // 将数据从主机复制到设备
    cudaMemcpy(d_positions, h_positions.data(), numParticles * 3 * sizeof(double), cudaMemcpyHostToDevice);
    // 设置性能标记
    nvtxRangePush("Force Computation");
    // 计算作用力
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    computeForces<<<numBlocks, blockSize>>>(d_positions, d_forces, numParticles);
    cudaDeviceSynchronize();
    // 结束性能标记
    nvtxRangePop();
    // 将结果从设备复制到主机
    cudaMemcpy(h_forces.data(), d_forces, numParticles * sizeof(double), cudaMemcpyDeviceToHost);
    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << "Force[" << i << "] = " << h_forces[i] << std::endl;
    }
    // 清理资源
    cudaFree(d_positions);
    cudaFree(d_forces);
    return 0;
}
