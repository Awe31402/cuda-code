// 05_gpu_load.cu
//
// 產生可控的 GPU 負載，用來觀察 P-State 怎麼隨負載變化。
// 給 06_pstate_monitor.sh 當作被觀測的對象。
//
// build: nvcc -arch=sm_89 -O2 -o 05_gpu_load 05_gpu_load.cu
// usage: ./05_gpu_load [秒數] [工作集MB]
//          ./05_gpu_load 10        -> 燒 10 秒，預設 16 MB 工作集（compute-bound）
//          ./05_gpu_load 10 2048   -> 燒 10 秒，2 GB 工作集（記憶體吃重）
//
// 兩種工作集是刻意的：可以觀察到同樣 100% 使用率下，
// 記憶體吃重的負載 SM 時脈更低（功耗預算被記憶體子系統分走）。

#include <cstdio>
#include <cstdlib>
#include <ctime>
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

__global__ void burn(float *o, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = o[i];
    for (int k = 0; k < 2048; ++k)
        x = fmaf(x, 1.0000001f, 0.5f);      // fmaf 讓編譯器無法最佳化掉迴圈
    o[i] = x;
}

static double now_sec() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    int    secs = (argc > 1) ? atoi(argv[1]) : 10;
    size_t mb   = (argc > 2) ? (size_t)atoll(argv[2]) : 16;

    size_t bytes = mb << 20;
    size_t n     = bytes / sizeof(float);

    float *d = nullptr;
    CHECK(cudaMalloc(&d, bytes));
    CHECK(cudaMemset(d, 0, bytes));

    printf("工作集 %zu MB，持續 %d 秒（另開一個終端跑 nvidia-smi 觀察）\n", mb, secs);
    fflush(stdout);

    int    grid = (int)((n + 255) / 256);
    double t0   = now_sec();
    do {
        for (int r = 0; r < 4; ++r)
            burn<<<grid, 256>>>(d, n);
        CHECK(cudaDeviceSynchronize());
    } while (now_sec() - t0 < secs);

    printf("結束（實際 %.1f 秒）\n", now_sec() - t0);
    CHECK(cudaFree(d));
    return 0;
}
