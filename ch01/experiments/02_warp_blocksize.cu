// 02_warp_blocksize.cu
//
// 實測「warp 沒填滿 32 個 thread」對效能的影響。
// ch01-questions.md 第 1-3 節那張表的數據來源。
//
// 實驗設計（重點：控制變因）
//   - 總 thread 數固定 4M，每個 thread 的運算量完全相同
//   - 只改 blockDim，grid 跟著調整，所以「總工作量」不變
//   - 因此時間差異只可能來自：warp 填不滿 + occupancy
//
// 預期會看到三件事
//   (1) blockDim=1  慢約 32 倍  -> 證明硬體以 32 lane 為單位發射
//   (2) blockDim=33 慢很多，但 blockDim=63 幾乎不慢
//       -> 懲罰來自「不滿的那個 warp 佔全體比例」，不是「是否為 32 的倍數」
//   (3) blockDim=32 就算 warp 全滿仍略慢於 64 -> 這是 occupancy 問題
//
// build: nvcc -arch=sm_89 -o 02_warp_blocksize 02_warp_blocksize.cu
// run  : ./02_warp_blocksize

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

// 每個 thread 做固定 256 次 FMA。
// 用 fmaf 而不是簡單加法，是為了讓編譯器沒辦法把迴圈最佳化掉，
// 並且讓 kernel 偏 compute-bound，凸顯發射槽的浪費。
__global__ void work(float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                        // 標準邊界檢查
    float x = out[i];
    for (int k = 0; k < 256; ++k)
        x = fmaf(x, 1.0000001f, 0.5f);
    out[i] = x;
}

static const int REPEAT = 20;

// 回傳單次 kernel 的平均毫秒數
static float timeIt(int block, int n, float *d) {
    int grid = (n + block - 1) / block;

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    work<<<grid, block>>>(d, n);               // warmup，排除首次啟動成本
    cudaDeviceSynchronize();

    cudaEventRecord(beg);
    for (int r = 0; r < REPEAT; ++r)
        work<<<grid, block>>>(d, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, beg, end);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return ms / REPEAT;
}

int main() {
    const int n = 1 << 22;                     // 4,194,304 個 thread
    const int WARP = 32;

    float *d = nullptr;
    CHECK(cudaMalloc(&d, (size_t)n * sizeof(float)));
    CHECK(cudaMemset(d, 0, (size_t)n * sizeof(float)));

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("device : %s (SM %d.%d, %d SMs, warpSize %d)\n",
           p.name, p.major, p.minor, p.multiProcessorCount, p.warpSize);
    printf("total threads = %d，每 thread 運算量相同，只改 blockDim\n\n", n);

    const int blocks[] = {1, 8, 16, 32, 33, 63, 64, 65, 96, 128, 129, 256};
    const int NB = sizeof(blocks) / sizeof(blocks[0]);

    // 先量基準（blockDim=64，warp 全滿且 occupancy 100%）
    float base = timeIt(64, n, d);

    printf("%-8s %-7s %-8s %-9s %-10s %-8s\n",
           "block", "warps", "lane槽", "lane效率", "ms", "vs 64");
    printf("-------- ------- -------- --------- ---------- --------\n");

    for (int i = 0; i < NB; ++i) {
        int b = blocks[i];
        int warps = (b + WARP - 1) / WARP;     // 無條件進位：不滿也要佔一整個 warp
        int slots = warps * WARP;              // 硬體實際配置的 lane 數
        float eff = 100.0f * b / slots;        // 有多少 lane 真的在做事
        float ms = timeIt(b, n, d);

        char effs[16];
        snprintf(effs, sizeof(effs), "%.1f%%", eff);

        printf("%-8d %-7d %-8d %-9s %-10.3f %.2fx%s\n",
               b, warps, slots, effs, ms, ms / base,
               (b % WARP) ? "   <- warp 沒填滿" : "");
    }

    printf("\n基準 blockDim=64 : %.3f ms\n", base);
    printf("\n對照 occupancy（本機每 SM 上限 %d blocks / %d threads）：\n",
           p.maxBlocksPerMultiProcessor, p.maxThreadsPerMultiProcessor);
    printf("  blockDim=32 -> %d blocks x 32 = %d threads，occupancy %.0f%%（撞 block 數上限）\n",
           p.maxBlocksPerMultiProcessor,
           p.maxBlocksPerMultiProcessor * 32,
           100.0 * p.maxBlocksPerMultiProcessor * 32 / p.maxThreadsPerMultiProcessor);
    printf("  blockDim=64 -> %d blocks x 64 = %d threads，occupancy 100%%\n",
           p.maxThreadsPerMultiProcessor / 64, p.maxThreadsPerMultiProcessor);

    CHECK(cudaFree(d));
    return 0;
}
