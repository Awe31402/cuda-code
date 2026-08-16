// 03_context_compare.cu
//
// 實測 Runtime API 的 primary context 與 Driver API 的 cuCtxCreate 有何不同。
// ch01-questions.md 第 2-3 節那段輸出的來源。
//
// 這支程式同時 include cuda.h（Driver API）和 cuda_runtime.h（Runtime API），
// 用 Driver API 當「觀察工具」去看 Runtime API 在背後做了什麼。
//
// 想證明四件事
//   (1) Runtime 的 context 是「延遲建立」的 —— 第一次 runtime 呼叫才生出來
//   (2) Runtime 用的是 primary context（每裝置唯一）
//   (3) cuCtxCreate 建的是「另一個」獨立 context，並 push 到 thread 的 context stack
//   (4) Runtime API 會跟隨 context stack 頂端 —— 兩種 API 可以混用
//
// 注意要多連結 -lcuda（Driver API 在 libcuda，由顯示卡驅動安裝，不在 Toolkit）
//
// build: nvcc -arch=sm_89 -o 03_context_compare 03_context_compare.cu -lcuda
// run  : ./03_context_compare

#include <cstdio>
#include <cuda.h>            // Driver API : cu*
#include <cuda_runtime.h>    // Runtime API: cuda*

#define DRV(call)                                                              \
    do {                                                                       \
        CUresult r = (call);                                                   \
        if (r != CUDA_SUCCESS) {                                               \
            const char *s = nullptr;                                           \
            cuGetErrorString(r, &s);                                           \
            printf("Driver API error %s:%d: %s\n", __FILE__, __LINE__, s);     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main() {
    CUcontext cur = nullptr;
    unsigned int flags = 0;
    int active = 0;

    // cuInit 只初始化 Driver API 本身，不會建立任何 context
    DRV(cuInit(0));
    CUdevice dev;
    DRV(cuDeviceGet(&dev, 0));

    // ---- (1) 還沒碰 Runtime API 之前 ----
    DRV(cuCtxGetCurrent(&cur));
    printf("[1] before any runtime call, current ctx = %p\n", (void *)cur);

    DRV(cuDevicePrimaryCtxGetState(dev, &flags, &active));
    printf("[2] primary ctx active? %d\n", active);
    printf("    -> current 是 NULL 且 primary 未啟動，證明 context 是延遲建立的\n\n");

    // ---- (2) 隨便一個 Runtime 呼叫就會建立並綁定 primary context ----
    // cudaFree(0) 是慣用手法：什麼都不做，純粹觸發 runtime 初始化
    cudaFree(0);
    DRV(cuCtxGetCurrent(&cur));
    DRV(cuDevicePrimaryCtxGetState(dev, &flags, &active));
    printf("[3] after cudaFree(0): current ctx = %p, primary active? %d\n",
           (void *)cur, active);
    printf("    -> 證明 Runtime API 隱式使用 primary context\n\n");

    CUcontext primary = cur;

    // ---- (3) cuCtxCreate 建一個全新的、獨立的 context ----
    // 建好後會自動 push 到「呼叫這個函式的 host thread」的 context stack 頂端
    CUcontext mine = nullptr;
    DRV(cuCtxCreate(&mine, 0, dev));
    DRV(cuCtxGetCurrent(&cur));
    printf("[4] after cuCtxCreate: new ctx = %p, current = %p (primary was %p)\n",
           (void *)mine, (void *)cur, (void *)primary);
    printf("    -> handle 與 primary 不同，證明 cuCtxCreate 建的是獨立 context\n");
    printf("    -> current 已變成新的那個，證明它被 push 到堆疊頂端\n\n");

    // ---- (4) 此時 Runtime API 會跟著 context stack 頂端走 ----
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, 1024);
    printf("[5] cudaMalloc inside cuCtxCreate ctx -> %s\n",
           (e == cudaSuccess) ? "OK (runtime followed the stack)"
                              : cudaGetErrorString(e));

    // 問驅動：這個指標到底屬於哪個 context？
    CUcontext owner = nullptr;
    DRV(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)p));
    printf("    cuPointerGetAttribute(CONTEXT) -> %p (%s)\n", (void *)owner,
           owner == mine ? "屬於 cuCtxCreate 的 ctx" : "屬於 primary");
    printf("    -> 兩種 API 可混用，且記憶體的歸屬是 driver ctx\n\n");

    // ---- (5) pop 掉之後回到 primary，證明真的是一個堆疊 ----
    DRV(cuCtxPopCurrent(&mine));
    DRV(cuCtxGetCurrent(&cur));
    printf("[6] after cuCtxPopCurrent: current = %p (back to primary %p)\n",
           (void *)cur, (void *)primary);
    printf("    -> 證明 context stack 的 push/pop 語意\n\n");

    // ---- (6) 歸屬 vs. 可存取性：這兩件事不一樣 ----
    // 常見說法是「A context 的指標在 B context 完全不能用」。
    // 但本機開著 UVA（Unified Virtual Addressing），同一張卡上跨 context 存取
    // 實際上是成功的。真正的分界線在「生命週期」，見 [8]。
    int uva = 0;
    DRV(cuDeviceGetAttribute(&uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev));
    e = cudaMemset(p, 0x11, 1024);
    cudaError_t sync1 = cudaDeviceSynchronize();
    printf("[7] UVA=%d，從 primary 存取 `mine` 的指標 -> memset=%s, sync=%s\n",
           uva, cudaGetErrorString(e), cudaGetErrorString(sync1));
    printf("    -> 有 UVA 時「跨 context 不能存取」其實不成立\n\n");
    cudaGetLastError();

    size_t free_b = 0, total_b = 0;
    DRV(cuMemGetInfo(&free_b, &total_b));
    printf("    mem free %.0f MB / %.0f MB\n\n", free_b / 1048576.0, total_b / 1048576.0);

    // ---- (7) 真正的差別：記憶體的生命週期綁在 context 上 ----
    // cuCtxCreate 出來的必須自己銷毀；primary context 則是驅動用參考計數管理。
    // context 一死，它擁有的所有配置跟著消失 —— 指標立刻變成野指標。
    printf(">>> cuCtxDestroy(mine)\n");
    DRV(cuCtxDestroy(mine));
    cudaGetLastError();

    e = cudaMemset(p, 0x22, 1024);
    printf("[8] 銷毀擁有者 context 後，同一個指標 -> %s\n", cudaGetErrorString(e));
    printf("    -> 記憶體隨 context 一起消失，這才是 context 隔離真正的意義\n");
    cudaGetLastError();

    printf("\n結論：\n");
    printf("  1. cuCtxCreate 建的是獨立 context，與 Runtime 用的 primary context 不同（見 [4]）\n");
    printf("  2. Runtime API 跟隨 context stack 頂端，兩者可混用（見 [5]）\n");
    printf("  3. 有 UVA 時跨 context 存取「可以」成功（見 [7]），但記憶體歸屬與\n");
    printf("     生命週期仍綁在建立它的 context 上（見 [8]）\n");
    printf("  4. 因此要與 Runtime API／cuBLAS 這類函式庫互通，建議用\n");
    printf("     cuDevicePrimaryCtxRetain 取得同一個 primary context，而非 cuCtxCreate\n");
    return 0;
}
