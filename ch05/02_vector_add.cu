// chapter5 片段 2（自 src/chapter5.txt 抽出）
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <ctime>
// 宏定义：通用错误检测与日志记录
#define CHECK_CUDA_ERROR(call)                                                       \
    {                                                                                \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess) {                                                    \
            std::ofstream logFile("cuda_error.log", std::ios::app);                  \
            time_t now = time(0);                                                    \
            char* dt = ctime(&now);                                                  \
            std::cerr << "CUDA错误: " << cudaGetErrorString(err)                     \
                      << " 在文件 " << __FILE__                                      \
                      << " 的第 " << __LINE__ << " 行" << std::endl;                \
            if (logFile.is_open()) {                                                 \
                logFile << "[" << dt << "] CUDA错误: " << cudaGetErrorString(err)    \
                        << " 在文件 " << __FILE__                                    \
                        << " 的第 " << __LINE__ << " 行" << std::endl;              \
                logFile.close();                                                     \
            }                                                                        \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    }
// 核函数：简单向量加法
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);
    std::cout << "主机内存分配中..." << std::endl;
    float *hostA = new float[N];
    float *hostB = new float[N];
    float *hostC = new float[N];
    for (int i = 0; i < N; ++i) {
        hostA[i] = static_cast<float>(i);
        hostB[i] = static_cast<float>(i * 2);
    }
    std::cout << "设备内存分配中..." << std::endl;
    float *deviceA, *deviceB, *deviceC;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceA, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&deviceB, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&deviceC, bytes));
    std::cout << "数据传输到设备中..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice));
    std::cout << "启动核函数进行计算..." << std::endl;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "从设备传输结果到主机中..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost));
    std::cout << "验证计算结果中..." << std::endl;
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (hostC[i] != hostA[i] + hostB[i]) {
            correct = false;
            break;
        }
    }
    std::cout << "结果验证: " << (correct ? "正确" : "错误") << std::endl;
    std::cout << "释放资源中..." << std::endl;
    CHECK_CUDA_ERROR(cudaFree(deviceA));
    CHECK_CUDA_ERROR(cudaFree(deviceB));
    CHECK_CUDA_ERROR(cudaFree(deviceC));
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    return 0;
}
