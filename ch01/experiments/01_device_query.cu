// 01_device_query.cu
//
// 查詢本機 GPU 的硬體規格。
// ch01-questions.md「本機實測環境」那張表的數據來源。
//
// 重點看三個數字：
//   multiProcessorCount          -> SM 數量（GPU 的「核心數」）
//   warpSize                     -> 固定 32，warp 機制的根本
//   maxThreadsPerMultiProcessor  -> 除以 32 就是每 SM 最大 warp 數（occupancy 分母）
//
// build: nvcc -o 01_device_query 01_device_query.cu
// run  : ./01_device_query

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(e));                                     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    printf("device count: %d\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        CHECK(cudaGetDeviceProperties(&p, i));

        printf("\n===== device %d: %s =====\n", i, p.name);

        printf("\n-- 架構 --\n");
        printf("compute capability        : %d.%d   (編譯用 -arch=sm_%d%d)\n",
               p.major, p.minor, p.major, p.minor);
        printf("clockRate                 : %.2f GHz\n", p.clockRate / 1e6);

        printf("\n-- SM 與 warp（第一題重點）--\n");
        printf("SM count                  : %d\n", p.multiProcessorCount);
        printf("warpSize                  : %d\n", p.warpSize);
        printf("maxThreadsPerBlock        : %d\n", p.maxThreadsPerBlock);
        printf("maxThreadsPerSM           : %d\n", p.maxThreadsPerMultiProcessor);
        printf("max warps per SM          : %d   (= %d / %d，occupancy 的分母)\n",
               p.maxThreadsPerMultiProcessor / p.warpSize,
               p.maxThreadsPerMultiProcessor, p.warpSize);
        printf("maxBlocksPerSM            : %d\n", p.maxBlocksPerMultiProcessor);

        // 這兩個上限「先撞到哪個算哪個」，是 blockDim 太小會掉 occupancy 的原因
        int by_block = p.maxBlocksPerMultiProcessor;
        printf("\n   blockDim 對 occupancy 的影響（兩個上限取小）：\n");
        for (int b = 32; b <= 256; b *= 2) {
            int blocks = p.maxThreadsPerMultiProcessor / b;
            if (blocks > by_block) blocks = by_block;
            int threads = blocks * b;
            double occ = 100.0 * threads / p.maxThreadsPerMultiProcessor;
            printf("   blockDim=%-4d -> %2d blocks, %4d threads, occupancy %3.0f%%%s\n",
                   b, blocks, threads, occ,
                   (blocks == by_block && occ < 100.0)
                       ? "  <- 撞到 block 數上限，occupancy 掉一半" : "");
        }

        printf("\n-- 記憶體 --\n");
        printf("globalMem                 : %.2f GB\n", p.totalGlobalMem / 1073741824.0);
        printf("regsPerBlock              : %d\n", p.regsPerBlock);
        printf("regsPerSM                 : %d\n", p.regsPerMultiprocessor);
        printf("sharedMemPerBlock         : %zu B (%zu KB)\n",
               p.sharedMemPerBlock, p.sharedMemPerBlock / 1024);
        printf("sharedMemPerSM            : %zu B (%zu KB)\n",
               p.sharedMemPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024);
        printf("l2CacheSize               : %d B (%d KB)\n",
               p.l2CacheSize, p.l2CacheSize / 1024);
        printf("memoryBusWidth            : %d bit\n", p.memoryBusWidth);
    }
    return 0;
}
