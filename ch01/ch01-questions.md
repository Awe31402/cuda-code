# 請解釋 SM 在 CUDA 架構中的作用，SM 如何透過 Warp 機制管理執行緒？ 如果一個 Warp 的執行緒數不足32個，會對性能產生怎樣的影響？

> 本文假設你熟 C/C++，但沒寫過 CUDA。所有 CUDA 概念都會先給一個 C/C++ 對照，術語整理在文末[附錄](#附錄cuda-術語表)。

## 配套程式碼

本文所有數據都不是抄規格書，是在本機跑出來的。程式碼在 [experiments/](experiments/)，可以直接重跑：

```bash
cd ch01/experiments
make run
```

| 程式 | 對應章節 | 在證明什麼 |
|---|---|---|
| [01_device_query.cu](experiments/01_device_query.cu) | [1-1](#1-1-sm-的作用) | 本機真實硬體規格：SM 數、warpSize、每 SM 上限、occupancy |
| [02_warp_blocksize.cu](experiments/02_warp_blocksize.cu) | [1-3](#1-3-warp-執行緒數不足-32-的影響) | warp 沒填滿 32 個 thread 會慢多少 |
| [03_context_compare.cu](experiments/03_context_compare.cu) | [2-3](#2-3-cuctxcreate-與-runtime-api-的-context-管理有何不同) | `cuCtxCreate` 與 primary context 是**不同**的東西 |
| [04_nvcc_pipeline.sh](experiments/04_nvcc_pipeline.sh) | [3](#三nvcc-的編譯流程目標碼類型與多架構設定) | nvcc 的每個編譯階段、PTX/SASS 差異、JIT fallback |
| [05_gpu_load.cu](experiments/05_gpu_load.cu) | [4](#四p-state) | 可控的 GPU 負載產生器（給 06 觀測用） |
| [06_pstate_monitor.sh](experiments/06_pstate_monitor.sh) | [4](#四p-state) | P-State 隨負載變化的即時取樣 |
| [07_nvidia_smi_query.sh](experiments/07_nvidia_smi_query.sh) | [5](#五用-nvidia-smi-查詢即時狀態) | nvidia-smi 各種查詢寫法 |

`make run` 跑前三支（很快）；`make run-tools` 跑後三支（06 會加負載，約一分鐘）。

讀法建議：**每一節先看結論，再對照該節引用的程式碼片段**，最後自己跑一次改參數玩玩看。

## 先破除一個最大的誤解

寫 C/C++ 的人看到「thread」會想到 `std::thread` / `pthread`：作業系統排程、有自己的堆疊、切換要存檔回復暫存器、開一個要幾微秒、開一千個機器就爆了。

**CUDA 的 thread 完全不是這種東西。**

CUDA thread 比較接近 **SIMD 的一個 lane（通道）**。你如果寫過 AVX，`__m512` 一次處理 16 個 float，那 16 個「格子」就是 lane。CUDA 的差別在於：它讓你**用寫純量程式的方式，寫 SIMD 程式**。

```c
// C 的寫法：迴圈
for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];

// AVX 的寫法：明確操作 16 個 lane
for (int i = 0; i < n; i += 16)
    _mm512_store_ps(&c[i], _mm512_add_ps(_mm512_load_ps(&a[i]),
                                          _mm512_load_ps(&b[i])));

// CUDA 的寫法：看起來是純量，但硬體是 SIMD
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 我是第幾個？
    if (i < n) c[i] = a[i] + b[i];                  // 就這樣
}
```

三個直接的後果：

1. **開 100 萬個 CUDA thread 是正常操作**，不是災難。它們幾乎不花錢。
2. **thread 切換是零成本的**（下面會解釋為什麼）。
3. **但它們不是獨立的**：32 個一組被綁在一起執行同一條指令。這一組就叫 **warp**，也是這題的重點。

## 本機實測環境

以下數據來自 [01_device_query.cu](experiments/01_device_query.cu)。整支程式的核心只有一個呼叫——`cudaGetDeviceProperties` 會把硬體規格填進一個結構：

```c
cudaDeviceProp p;
cudaGetDeviceProperties(&p, i);

printf("SM count        : %d\n", p.multiProcessorCount);        // 24
printf("warpSize        : %d\n", p.warpSize);                   // 32
printf("maxThreadsPerSM : %d\n", p.maxThreadsPerMultiProcessor); // 1536
printf("max warps per SM: %d\n", p.maxThreadsPerMultiProcessor / p.warpSize); // 48
```

`cudaDeviceProp` 就是個普通的 struct，欄位名稱查 CUDA 文件即可。跑 `./01_device_query` 得到：

| 項目 | 值 | C/C++ 心智對照 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | — |
| Compute Capability | 8.9 (Ada Lovelace) | 類似 `-march=x86-64-v3`，決定能用哪些指令 |
| **SM 數量** | **24** | 類似「CPU 有 24 個核心」 |
| **warpSize** | **32** | 類似「SIMD 寬度是 32 lane」 |
| 每個 SM 最大執行緒數 | 1536 | 類似「每核心可以有 1536-way SMT」 |
| 每個 SM 最大 warp 數 | 48 | 1536 ÷ 32 |
| 每個 SM 最大 block 數 | 24 | — |
| 每個 block 最大執行緒數 | 1024 | — |
| 每個 SM 暫存器 | 65536 個 32-bit | 巨大的暫存器檔案，是零成本切換的關鍵 |
| 每個 SM 共享記憶體 | 102400 B (100 KB) | 類似「可以自己控制的 L1」 |
| 每個 block 共享記憶體 | 49152 B (48 KB) | — |
| Driver / Toolkit | 580.167.08 (支援 CUDA 13.0) / nvcc 12.4 | 類似「核心版本 vs. glibc 版本」 |

## 1-1 SM 的作用

**SM（Streaming Multiprocessor，串流多處理器）就是 GPU 的「核心」。** 一顆 GPU 由多個 SM 組成，本機是 24 個。你的程式碼實際上是在 SM 裡面跑的。

SM 內部有什麼（對照 CPU 核心來看）：

| SM 元件 | 作用 | CPU 對照 |
|---|---|---|
| CUDA Core / FP32 單元 | 做浮點與整數運算 | ALU / FPU |
| **Warp Scheduler** | 每個 cycle 決定要發射哪個 warp 的指令 | 指令排程器，但這裡是「選一組 32 個」 |
| **Register File**（64K 個暫存器） | 存所有駐留執行緒的變數 | 暫存器，但大了好幾個數量級 |
| **Shared Memory**（100 KB） | 同 block 執行緒共用的高速暫存區 | **可程式控制的 L1**（CPU 沒有對應物） |
| L1 Cache | 快取 | L1 |
| LD/ST 單元 | 記憶體讀寫 | Load/Store unit |
| SFU | 算 `sin`/`cos`/`sqrt`/`exp` 等 | 沒有直接對應，CPU 用軟體算 |
| Tensor Core | 專門做矩陣乘加（AI 用） | 有點像 AMX |

### 工作是怎麼被分派下去的

CUDA 的執行單位有四層。C/C++ 沒有完全對應的東西，但可以這樣想：

```
Kernel（你寫的 __global__ 函式）
  └─ Grid（這次啟動的全部執行緒，像一個超大的 parallel_for 範圍）
       └─ Block（一群執行緒，會被整組丟到「某一個」SM 上）
            └─ Warp（32 個 thread，硬體排程的最小單位）← 重點
                 └─ Thread（一個 SIMD lane）
```

你在程式裡只需要決定兩件事，就是 `<<<>>>` 裡的兩個數字：

```c
add<<<grid, block>>>(a, b, c, n);
//    ↑      ↑
//    |      └─ 每個 block 有幾個 thread（blockDim），例如 256
//    └─ 總共有幾個 block（gridDim），例如 (n + 255) / 256
```

`warp` 這一層**你不能控制，也看不到**——它是硬體自動切出來的。但它決定了效能。

### 四條必須記住的規則

1. **一個 block 只會落在一個 SM 上**，不會被拆開跨 SM。
   → 這就是為什麼「同 block 的執行緒才能用 shared memory 和 `__syncthreads()`」。它們保證在同一塊硬體上。
2. **一個 SM 可以同時放很多個 block**（本機上限：24 個 block，或 1536 個執行緒，先撞到哪個算哪個）。
3. **Block 分派到哪個 SM、誰先跑完，完全由硬體決定。** 你的程式**絕對不可以**假設 block 之間的執行順序。這跟 C++ 沒有 memory ordering 保證時的思維類似，但更嚴格——block 之間連同步的手段都沒有。
4. **Block 一旦進了 SM 就常駐到結束**（resident），中間不會被搬走或換出。跟 OS thread 會被 preempt 完全不同。

### Occupancy：GPU 效能的核心

```
Occupancy（佔用率） = SM 上實際駐留的 warp 數 ÷ 硬體上限（本機 48）
```

為什麼重要？因為 **GPU 不靠「跑得快」贏，靠「等的時候有別的事做」贏**。

存取一次 global memory 要 ~400-800 個 cycle。CPU 的解法是蓋一個巨大的 cache 加上亂序執行。GPU 的解法很暴力：

> 這個 warp 在等記憶體？沒關係，我還有 47 個 warp，換下一個。

這叫 **latency hiding（延遲隱藏）**。而且切換是**零成本**的——因為所有駐留 warp 的暫存器**全都同時待在那 64K 個暫存器裡**，不需要像 CPU context switch 那樣存檔／回復。這就是為什麼 SM 的 register file 要做得那麼大。

所以 occupancy 太低 = 沒有備胎可換 = SM 在乾等。

### 用程式算 occupancy

[01_device_query.cu](experiments/01_device_query.cu) 裡有這段。重點是**兩個上限先撞到哪個算哪個**（`if (blocks > by_block) blocks = by_block;`）：

```c
int by_block = p.maxBlocksPerMultiProcessor;         // 24
for (int b = 32; b <= 256; b *= 2) {
    int blocks = p.maxThreadsPerMultiProcessor / b;  // 依 thread 數上限能放幾個 block
    if (blocks > by_block) blocks = by_block;        // ← 但不能超過 block 數上限
    int threads = blocks * b;
    // occupancy = threads / maxThreadsPerSM
}
```

實際輸出：

```
blockDim=32   -> 24 blocks,  768 threads, occupancy  50%  <- 撞到 block 數上限，occupancy 掉一半
blockDim=64   -> 24 blocks, 1536 threads, occupancy 100%
blockDim=128  -> 12 blocks, 1536 threads, occupancy 100%
blockDim=256  ->  6 blocks, 1536 threads, occupancy 100%
```

`blockDim=32` 明明 warp 是滿的，occupancy 卻只有 50%——因為 1536/32 = 48 個 block，但硬體最多只收 24 個。**這個伏筆在 1-3 的實測時間裡會被證實。**

## 1-2 Warp 機制如何管理執行緒

**一句話：Warp 是 SM 排程的最小單位，固定 32 個執行緒。** 本機實測 `warpSize = 32`（至今所有 NVIDIA GPU 都是 32）。

### Warp 是怎麼切出來的

Block 進入 SM 後，硬體會照 `threadIdx` 的**線性順序**，每 32 個切一個 warp：

- 一維：thread 0~31 → warp 0，thread 32~63 → warp 1，依此類推
- 多維：先攤平成一維再切（x 變化最快，跟 C 的 row-major 陣列一樣）
  ```c
  tid = threadIdx.x
      + threadIdx.y * blockDim.x
      + threadIdx.z * blockDim.x * blockDim.y;
  ```

### SIMT：這才是關鍵

**SIMT = Single Instruction, Multiple Threads**。NVIDIA 造的詞，本質上就是 **SIMD 加上「每個 lane 可以被獨立遮罩關掉」**。

同一個 warp 的 32 個 thread，**每個 cycle 執行同一條指令**，只是各自作用在不同資料上。它們（在 Volta 之前）共用一個 Program Counter。

```
warp 內的 32 個 thread：
  指令流:  LOAD → LOAD → FADD → STORE     ← 只有一份指令
  資料:    lane0 lane1 lane2 ... lane31   ← 32 份不同的資料
```

補充兩點：

- Ada 架構的 SM 內部分成 **4 個 processing block（子分區）**，每個有自己的 warp scheduler。所以一個 SM 每 cycle 可以同時發射 4 個 warp 的指令。
- Volta 之後每個 thread 有獨立的 PC（叫 Independent Thread Scheduling），但**排程仍然以 warp 為單位**，本題的結論不受影響。

### Warp Divergence（執行緒分歧）

這是 SIMT 的代價，也是 C/C++ 轉 CUDA 最容易踩的坑。

因為 32 個 lane 共用一條指令流，**如果它們想走不同的 if 分支，硬體只能序列化執行**：

```c
if (threadIdx.x % 2 == 0) {
    A();      // 硬體：跑 A，把奇數 lane 遮罩關掉（它們空轉）
} else {
    B();      // 硬體：跑 B，把偶數 lane 遮罩關掉（它們空轉）
}
// 總時間 = time(A) + time(B)，不是 max(A, B)
```

那些被關掉的 lane 叫做 **inactive**，由 **active mask**（一個 32-bit 遮罩）控制。這跟 AVX-512 的 mask register 是同一個概念。

**關鍵：分歧只在 warp 內部才有代價。** 不同 warp 走不同分支完全沒問題——它們本來就是分開排程的。

```c
if (threadIdx.x < 32) { A(); } else { B(); }   // 沒有分歧！剛好切在 warp 邊界
if (threadIdx.x % 2)  { A(); } else { B(); }   // 最糟：每個 warp 都分歧
```

本章的 [01_branch_optimization.cu](01_branch_optimization.cu) 和 [03_warp_execution.cu](03_warp_execution.cu) 就是在示範這件事。

## 1-3 Warp 執行緒數不足 32 的影響

### 先講機制

**硬體永遠以 32 個 lane 為單位配置與發射，沒有例外。**

如果你的 block 只有 8 個 thread，硬體**還是配一個完整的 warp**，只是把其中 24 個 lane 用 active mask 關掉。那 24 個 lane 照樣佔著發射槽（issue slot）、照樣消耗一個 cycle，但**不做任何有用的事**。

用 C 的比喻：像是你呼叫 `_mm512_add_ps`（一次算 16 個 float），但只有 1 個格子放了有效資料——指令的成本一毛都沒少。

### 實驗設計：[02_warp_blocksize.cu](experiments/02_warp_blocksize.cu)

要證明「warp 沒填滿會變慢」，關鍵是**控制變因**——時間差異必須只可能來自 warp，不能來自別的東西。

**第一步：讓每個 thread 做完全一樣的工作。**

```c
__global__ void work(float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                        // 標準邊界檢查
    float x = out[i];
    for (int k = 0; k < 256; ++k)
        x = fmaf(x, 1.0000001f, 0.5f);         // 固定 256 次 FMA
    out[i] = x;
}
```

兩個刻意的選擇：

- 用 `fmaf` 而不是簡單加法 — 讓編譯器**沒辦法把迴圈最佳化掉**。如果寫 `x += 1.0f`，`-O2` 會直接算成 `x + 256` 一步解決，整個實驗就毀了。
- 迴圈 256 次讓 kernel 偏 **compute-bound**（瓶頸在算，不在搬資料）。這樣才凸顯得出「發射槽被浪費」；如果是 memory-bound，時間會被記憶體延遲蓋過去。

**第二步：固定總工作量，只改 blockDim。**

```c
const int n = 1 << 22;                  // 4,194,304 個 thread，固定不變
int grid = (n + block - 1) / block;      // grid 跟著 block 反向調整
work<<<grid, block>>>(d, n);
```

`grid × block ≈ n` 恆成立，所以不管 blockDim 怎麼變，**總共就是 4M 個 thread 做 4M 份相同的工作**。時間變了，就只能是 warp 或 occupancy 造成的。

**第三步：正確計時。**

```c
work<<<grid, block>>>(d, n);        // warmup：排除首次啟動的一次性成本
cudaDeviceSynchronize();

cudaEventRecord(beg);
for (int r = 0; r < REPEAT; ++r)    // 跑 20 次取平均
    work<<<grid, block>>>(d, n);
cudaEventRecord(end);
cudaEventSynchronize(end);          // ← 一定要同步，否則量到的是「送出指令」的時間

float ms;
cudaEventElapsedTime(&ms, beg, end);
return ms / REPEAT;
```

這裡有兩個 C/C++ 開發者一定會踩的坑：

- **不能用 `clock()` 或 `std::chrono` 直接量**。kernel 啟動是非同步的，`work<<<>>>()` 立刻就返回了，你會量到接近 0。要用 `cudaEvent`（插在 GPU 命令佇列裡的時間戳），或至少先 `cudaDeviceSynchronize()`。
- **一定要 warmup**。第一次啟動 kernel 包含 context 建立、模組載入等一次性成本，會嚴重汙染數據。

**第四步：把「佔幾個 warp」算出來一起印，讓表格自己解釋。**

```c
int warps = (b + WARP - 1) / WARP;   // 無條件進位 ← 這行就是整題的答案
int slots = warps * WARP;            // 硬體實際配置的 lane 數
float eff  = 100.0f * b / slots;     // 有多少 lane 真的在做事
```

`(b + 31) / 32` 這個無條件進位，就是「**不滿 32 也要佔一整個 warp**」的程式碼化身。blockDim=33 算出來是 2 個 warp、64 個 lane 槽，但只有 33 個在做事。

### 實測結果

`./02_warp_blocksize` 的實際輸出：

```
block    warps   lane槽  lane效率  ms         vs 64
1        1       32      3.1%      5.530      33.53x   <- warp 沒填滿
8        1       32      25.0%     0.701       4.25x   <- warp 沒填滿
16       1       32      50.0%     0.352       2.13x   <- warp 沒填滿
32       1       32      100.0%    0.177       1.07x
33       2       64      51.6%     0.312       1.89x   <- warp 沒填滿
63       2       64      98.4%     0.167       1.01x   <- warp 沒填滿
64       2       64      100.0%    0.165       1.00x
65       3       96      67.7%     0.232       1.40x   <- warp 沒填滿
96       3       96      100.0%    0.162       0.98x
128      4       128     100.0%    0.165       1.00x
129      5       160     80.6%     0.195       1.18x   <- warp 沒填滿
256      8       256     100.0%    0.166       1.01x
```

注意 **lane 效率那一欄和 ms 那一欄幾乎是反比**——這就是全部的答案。

### 三個從數據直接看出來的結論

**(1) blockDim=1 的浪費幾乎剛好是 32 倍 — 機制被實測證實**

理論：1 個 thread 佔滿 1 個 warp 的 32 個發射槽 → lane 效率 1/32 ≈ **3.1%**（程式印出來的就是這個數字）。

實測：拿同樣 warp 全滿、但沒有 occupancy 干擾的 `blockDim=32` 當基準，5.530 / 0.177 = **31.2 倍**。

跟理論值 32 幾乎完全吻合。這直接證明了「硬體以 32 lane 為單位發射」不是說法，是事實。

> 表格裡的 `33.53x` 是對 `blockDim=64` 的比值（程式的統一基準）。要單獨檢驗「32 lane」這個機制，用 `blockDim=32` 當基準才乾淨——因為 32 和 1 的 occupancy 情況相同，變因只剩 lane 效率。

**(2) 真正的懲罰來自「不滿的那個 warp 佔全體的比例」，不是「是不是 32 的倍數」**

這點跟很多教科書講的「blockDim 一定要是 32 的倍數」不太一樣，數據講得更精準。看 33 和 63，兩個**都不是** 32 的倍數：

| block | 需要幾個 warp | lane 槽總數 | 真正在做事 | lane 效率 | 實測 |
|---|---|---|---|---|---|
| 33 | 2 | 64 | 33 | **51.6%** | 慢 1.89 倍 |
| 63 | 2 | 64 | 63 | **98.4%** | 慢 1.01 倍（等於沒差） |

同樣的規律繼續成立：

- `block=65` → 3 個 warp、96 個 lane 槽裝 65 個 → 效率 67.7% → 實測慢 1.41 倍
- `block=129` → 5 個 warp、160 個 lane 槽裝 129 個 → 效率 80.6% → 實測慢 1.18 倍

**結論：block 越大，那一個不滿的 warp 就被攤薄得越多，懲罰越小。**

**(3) 就算 warp 全滿，block=32 仍然略慢於 block=64 — 這是 occupancy 問題**

這題暴露了另一個獨立的坑。本機每個 SM 有兩個上限，**先撞到哪個算哪個**：最多 24 個 block、最多 1536 個 thread。

| block size | 撞到哪個上限 | 實際駐留 thread | Occupancy |
|---|---|---|---|
| 32 | **24 個 block** | 24 × 32 = 768 | **50%** |
| 64 | 1536 個 thread | 24 × 64 = 1536 | **100%** |

block=32 的 warp 明明是滿的，但 SM 上只有 24 個 warp 可以拿來隱藏延遲（上限 48）。這解釋了實測中 block=32 (0.177 ms) 比 block=64 (0.165 ms) 慢約 7%。

[02_warp_blocksize.cu](experiments/02_warp_blocksize.cu) 最後會把這個對照直接印出來，不用自己心算：

```
對照 occupancy（本機每 SM 上限 24 blocks / 1536 threads）：
  blockDim=32 -> 24 blocks x 32 = 768 threads，occupancy 50%（撞 block 數上限）
  blockDim=64 -> 24 blocks x 64 = 1536 threads，occupancy 100%
```

**這是本題最容易被忽略的一點：「warp 填滿」和「occupancy 夠高」是兩件獨立的事，都要顧。**

### 想自己驗證？改這幾個地方

[02_warp_blocksize.cu](experiments/02_warp_blocksize.cu) 很短，建議動手改：

| 改什麼 | 怎麼改 | 預期會看到 |
|---|---|---|
| 測更多 blockDim | 改 `blocks[]` 陣列，加入 97、160、192 | 規律一致：lane 效率決定慢多少 |
| 證明 fmaf 是必要的 | 把 `x = fmaf(x, 1.0000001f, 0.5f)` 換成 `x += 1.0f` | 所有 blockDim 時間都變超短且趨近 — 迴圈被最佳化掉了，實驗失效 |
| 看 memory-bound 的差異 | 把迴圈次數 256 改成 1 | 差距明顯縮小 — 瓶頸變成記憶體，發射槽浪費就不那麼致命 |
| 證明 warmup 有必要 | 註解掉 warmup 那兩行 | 第一組數據（blockDim=1）會被一次性成本汙染 |

### 實務建議（可以直接照做）

1. **blockDim 取 32 的倍數，預設用 128 或 256。** 本機上 64/96/128/256 表現幾乎一樣（0.162~0.166 ms），沒必要糾結。
2. **絕對不要用 blockDim < 32。** 那是直接損失一個數量級。
3. **資料量不是 32 的倍數時，不要去調整 blockDim**，正常做法是加邊界檢查，讓最後一個 block 有些 thread 閒置：
   ```c
   __global__ void kernel(float *data, int n) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i >= n) return;        // ← 標準寫法，這點浪費可忽略
       // ...
   }
   ```
   從 block=63 的數據可以看到，只要 grid 夠大，這種零頭完全不影響效能。
4. **blockDim 也不能太小**，會撞到「每 SM 最大 block 數」的天花板，害 occupancy 掉一半（見結論 3）。

---

# CUDA 運行時 API 和驅動 API 的主要區別是什麼？在開發時如何選擇使用這兩種 API? 請說明驅動 API 中 cuCtxCreate 與運行時 API 中設備上下文管理方式有何不同？

## 一句話版本（給 C/C++ 開發者）

> **Runtime API 像「靜態連結 + 直接呼叫函式」，Driver API 像「`dlopen` + `dlsym` + 手動管理一切」。**

這個類比幾乎是字面上成立的，等一下會看到。

## 2-1 主要區別

| 面向 | Runtime API | Driver API |
|---|---|---|
| 標頭檔 | `cuda_runtime.h` | `cuda.h` |
| 連結函式庫 | `libcudart`（隨 **Toolkit** 發布，你要自己帶） | `libcuda`（隨**顯示卡驅動**安裝，系統本來就有） |
| 函式前綴 | `cuda*`：`cudaMalloc`、`cudaMemcpy` | `cu*`：`cuMemAlloc`、`cuMemcpyHtoD` |
| 初始化 | 隱式、第一次呼叫時自動做 | **必須**先呼叫 `cuInit(0)` |
| Context 管理 | 自動（用 primary context） | 手動 `cuCtxCreate` / `cuDevicePrimaryCtxRetain` |
| Kernel 載入 | 編譯期綁好，自動載入 | 手動 `cuModuleLoad` 載入 PTX/cubin |
| Kernel 啟動 | `kernel<<<grid, block>>>(args)` | `cuLaunchKernel(f, ..., void **args)` |
| 需要 nvcc | **是**（`<<<>>>` 只有 nvcc 認得） | **否**，純 gcc/clang/MSVC 就能編 |
| 錯誤型別 | `cudaError_t` | `CUresult` |
| 控制粒度 | 粗、方便 | 細、囉唆 |

### 同一件事的兩種寫法

啟動 kernel，Runtime API：

```c
// 編譯期就知道 vecAdd 是誰，nvcc 幫你把一切接好
vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

Driver API：

```c
cuInit(0);
cuDeviceGet(&dev, 0);
cuCtxCreate(&ctx, 0, dev);

CUmodule mod;  CUfunction f;
cuModuleLoad(&mod, "vecadd.ptx");           // ≈ dlopen("libfoo.so")
cuModuleGetFunction(&f, mod, "vecAdd");     // ≈ dlsym(handle, "vecAdd")

void *args[] = { &d_a, &d_b, &d_c, &n };    // 參數用 void** 陣列傳，沒有型別檢查
cuLaunchKernel(f, blocks,1,1, threads,1,1, 0, NULL, args, NULL);
```

看出來了嗎？`cuModuleLoad` / `cuModuleGetFunction` 就是 GPU 版的 `dlopen` / `dlsym`。連「參數用 `void**` 傳、沒有型別安全」這種不舒服的地方都一模一樣。

### 兩者不是互斥的

**Runtime API 底層本來就是在呼叫 Driver API**，而且可以在同一個程式裡混用（2-3 會用實測證明）。

本章的 [06_driver_load_ptx.cu](06_driver_load_ptx.cu) 和 [08_runtime_vs_driver.cu](08_runtime_vs_driver.cu) 就是這兩種寫法的對照範例。而 [experiments/03_context_compare.cu](experiments/03_context_compare.cu) 更進一步——它在**同一支程式裡同時 include 兩邊的標頭檔**，用 Driver API 當「觀察工具」去看 Runtime API 背後做了什麼：

```c
#include <cuda.h>            // Driver API : cu*
#include <cuda_runtime.h>    // Runtime API: cuda*
```

這在 2-3 節會派上用場。

### 一個容易被忽略的實務差異：版本相容

本機情況：driver 是 580.167.08（支援到 CUDA 13.0），但 nvcc 只有 12.4。

- **Driver API 有向後相容保證**：新驅動一定跑得動舊程式。
- **Runtime API** 則需要把對應版本的 `libcudart` 一起發布給使用者。

心智對照：`libcuda` 像 Linux 的 syscall 介面（核心保證不破壞相容），`libcudart` 像你自己 bundle 的一個 shared library。

## 2-2 開發時如何選擇

**預設選 Runtime API。** 除非你有下面「Driver API」清單裡的具體需求，否則不用考慮。

**用 Runtime API：**

- 一般應用程式、研究程式碼、學習用途（你現在的情況）
- kernel 在編譯期就確定
- 想用 `<<<>>>` 語法、想少寫 80% 的樣板碼
- 要用 cuBLAS / cuDNN / cuFFT / Thrust 等函式庫（它們都是 Runtime API 介面）

**用 Driver API（有具體需求才用）：**

- **執行期才決定要跑哪個 kernel**：從檔案或字串載入 PTX/cubin 做 JIT。深度學習框架、編譯器後端（PyTorch inductor、Triton、TVM）都靠這個。
- **不想依賴 nvcc**：例如做 Python / Rust / Java 的 CUDA 綁定，或 host 端要用 MSVC/clang 編。
- **需要多 context 的精細隔離**：同一張卡上要多個彼此獨立的位址空間。
- **要精準控制模組載入／卸載時機**，管理記憶體佔用。
- 寫的是驅動層工具：profiler、容器 runtime、監控程式。

實務上最常見的組合：**Runtime API 寫主體，只在需要動態載入 kernel 的地方切到 Driver API。**

## 2-3 cuCtxCreate 與 Runtime API 的 context 管理有何不同

### 先搞懂 Context 是什麼

> **CUDA context ≈ GPU 上的一個「行程（process）」。**

它擁有自己的：虛擬位址空間、記憶體配置、已載入的模組、streams、events。

最重要的後果，跟 OS 行程一樣：**記憶體的歸屬與生命週期綁在 context 上。context 一死，它擁有的所有配置跟著消失。**

> **注意一個常見的過度簡化。** 很多教材說「A context 的指標在 B context 裡完全不能用」。我實測發現**這在現代 GPU 上不成立**——本機開著 UVA（Unified Virtual Addressing，統一虛擬定址），同一張卡上跨 context 存取是**會成功**的。真正的分界線是**生命週期**，不是存取權。細節見下面的[證明 (4)(5)](#實測驗證03_context_comparecu)，兩種情況我都跑過。

### Runtime API 的做法：Primary Context

- 每個裝置**有且只有一個** primary context，由驅動管理。像個 singleton。
- **延遲建立（lazy）**：第一次呼叫任何 runtime API（例如 `cudaFree(0)`、`cudaMalloc`）時才建立。跟 C++11 的 function-local static 初始化時機很像。
- **參考計數**：同一個行程內，**所有 host thread 共用同一個** primary context。
  → 這是 Runtime API 好用的關鍵：你開 10 個 `std::thread`，它們都直接看得到同一批 `cudaMalloc` 出來的指標，不用做任何事。
- 你**拿不到也不需要** context handle。`cudaSetDevice(i)` 做的事就是「切到裝置 i + 綁定它的 primary context」。
- Driver API 這邊對應的操作是 `cuDevicePrimaryCtxRetain` / `cuDevicePrimaryCtxRelease`。

### Driver API 的 cuCtxCreate

- 建立一個**全新的、獨立的** context —— **不是** primary context，是另外一個。
- 建立後自動 **push 到「呼叫這個函式的那個 host thread」的 context stack 頂端**，成為 current context。
- **Context stack 是 per-host-thread 的**（thread-local，就像 `thread_local` 變數）。別的 thread 看不到你 push 的 context，除非明確用 `cuCtxSetCurrent` 綁上去。
- 必須自己 `cuCtxDestroy` 釋放。忘了就漏（RAII 在這裡幫不了你，除非自己包一層）。
- 同一張卡上可以有多個 `cuCtxCreate` 的 context，記憶體完全隔離；GPU 在它們之間切換**有實際成本**。

### 實測驗證：[03_context_compare.cu](experiments/03_context_compare.cu)

上面全是「說法」。這支程式用 Driver API 的三個**唯讀查詢函式**當觀察工具，把 Runtime API 背後的動作抓出來看：

| 觀察工具 | 作用 |
|---|---|
| `cuCtxGetCurrent(&cur)` | 現在 context stack 頂端是誰（NULL 代表沒有） |
| `cuDevicePrimaryCtxGetState(dev, &flags, &active)` | primary context 啟動了沒 |
| 比對兩個 `CUcontext` 的位址 | 判斷是不是同一個 context |

以下逐段對照「程式碼 → 輸出 → 證明了什麼」。

---

**證明 (1)：Runtime 的 context 是延遲建立的**

```c
cuInit(0);                       // 只初始化 Driver API 本身，不建立任何 context
cuDeviceGet(&dev, 0);

cuCtxGetCurrent(&cur);
cuDevicePrimaryCtxGetState(dev, &flags, &active);
```

```
[1] before any runtime call, current ctx = (nil)
[2] primary ctx active? 0
```

current 是 `NULL`、primary 也還沒啟動 → **光是 `cuInit` 不會生出 context**。這跟 C++11 的 function-local static 一樣，用到才初始化。

---

**證明 (2)：Runtime API 隱式使用 primary context**

```c
cudaFree(0);                     // ← 慣用手法：什麼都不做，純粹觸發 runtime 初始化
cuCtxGetCurrent(&cur);
cuDevicePrimaryCtxGetState(dev, &flags, &active);
CUcontext primary = cur;         // 記下來，後面要比對
```

```
[3] after cudaFree(0): current ctx = 0x55a05ec5d1b0, primary active? 1
```

只呼叫了一個什麼都不做的 `cudaFree(0)`，primary context 就**被建立並綁定成 current** → 證明 Runtime API 用的就是 primary context，而且是自動的。

> `cudaFree(0)` 是 CUDA 圈的老慣用法：想在計時前先把初始化成本付掉，就呼叫它。

---

**證明 (3)：cuCtxCreate 建的是「另一個」context — 本題核心**

```c
CUcontext mine;
cuCtxCreate(&mine, 0, dev);      // 建新 context，並自動 push 到堆疊頂端
cuCtxGetCurrent(&cur);
```

```
[4] after cuCtxCreate: new ctx = 0x55a05f605360, current = 0x55a05f605360 (primary was 0x55a05ec5d1b0)
```

看位址：

- `mine` = `0x...605360`
- `primary` = `0x...c5d1b0`
- **兩者不同** → `cuCtxCreate` 建的**不是** primary context，是全新獨立的一個
- 而且 `current` 已經變成 `mine` → 它被 **push 到 context stack 頂端**

這就是這題要的答案，而且是位址比對出來的，不是背下來的。

---

**證明 (4)：Runtime API 會跟隨 context stack 頂端，記憶體歸屬則是 driver ctx**

```c
void *p;
cudaMalloc(&p, 1024);            // Runtime API，但現在 current 是 driver 建的 ctx

CUcontext owner;                 // 直接問驅動：這指標屬於誰？
cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)p);
```

```
[5] cudaMalloc inside cuCtxCreate ctx -> OK (runtime followed the stack)
    cuPointerGetAttribute(CONTEXT) -> 0x56406b7d9360 (屬於 cuCtxCreate 的 ctx)
```

兩件事：

- `cudaMalloc` 沒報錯 → Runtime API **不是死綁 primary context，而是看 stack 頂端是誰就用誰**，兩種 API 可以混用。
- `cuPointerGetAttribute` 回報的 owner 正是 `mine` 的位址（`0x...7d9360`）→ 這塊記憶體**歸屬於 driver context**，不是 primary。

---

**證明 (5)：context stack 真的是一個堆疊**

```c
cuCtxPopCurrent(&mine);          // pop 掉頂端
cuCtxGetCurrent(&cur);
```

```
[6] after cuCtxPopCurrent: current = 0x56406ae311b0 (back to primary 0x56406ae311b0)
```

pop 之後 current 回到 `primary` 的位址 → 確認是 push/pop 語意的堆疊，而且這個堆疊是 **per-host-thread**（thread-local）的。

---

**證明 (6)：有 UVA 時，跨 context 存取其實是成功的**

這是我原本寫錯、被實測打臉的一點，值得單獨拉出來講。

現在 current 是 primary，但 `p` 屬於 `mine`。按教科書說法這應該要失敗：

```c
int uva;
cuDeviceGetAttribute(&uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
cudaMemset(p, 0x11, 1024);       // 從 primary 動 `mine` 的記憶體
cudaDeviceSynchronize();
```

```
[7] UVA=1，從 primary 存取 `mine` 的指標 -> memset=no error, sync=no error
```

**成功了。** 我另外測過更嚴格的情況——在 primary 裡啟動一個 kernel 去寫 `p`，然後 memcpy 回主機檢查內容——同樣成功，資料真的寫進去了。

原因是 **UVA（Unified Virtual Addressing）**：CUDA 4.0 之後，同一行程內所有 context 與主機共用一個虛擬位址空間，驅動知道每個位址屬於哪塊配置，因此能自動解析。所以「跨 context 指標完全不能用」這個說法，在現代 GPU 上是**過時的**。

---

**證明 (7)：真正的分界線是生命週期**

那 context 隔離到底隔離了什麼？答案是**擁有權**。銷毀 `mine`，它擁有的配置就跟著死：

```c
cuCtxDestroy(mine);              // cuCtxCreate 出來的必須自己銷毀，忘了就漏
cudaMemset(p, 0x22, 1024);       // 同一個指標，再動一次
```

```
>>> cuCtxDestroy(mine)
[8] 銷毀擁有者 context 後，同一個指標 -> an illegal memory access was encountered
```

**這才是 context 隔離真正的意義**：不是「別的 context 碰不到」，而是「**這塊記憶體的命跟著 context 走**」。context 一死，指標立刻變野指標。

用 C 的比喻：`p` 不是被權限擋住，而是像指向一個已經 `free()` 掉的 arena——你之前拿得到，是因為 arena 還活著。

而 primary context 不會有這個問題，因為它由驅動用**參考計數**管理，不會被你手滑銷毀。這就是 Driver API 囉唆但可控的代價——`malloc`/`free` 對比 `std::vector`。

### 對照總表

| | Runtime API | `cuCtxCreate` |
|---|---|---|
| Context 來源 | Primary context（每裝置唯一，singleton） | 每次呼叫都建一個新的 |
| 建立時機 | 第一次 runtime 呼叫時自動 | 你明確呼叫時 |
| 生命週期 | 參考計數，驅動管理 | 手動 `cuCtxDestroy`，忘了就漏 |
| 跨 host thread | **所有 thread 自動共用** | 只綁定呼叫的那個 thread（stack 是 thread-local） |
| 拿得到 handle 嗎 | 拿不到，也不需要 | 直接持有 `CUcontext` |
| 怎麼切換 | `cudaSetDevice` | `cuCtxPushCurrent` / `cuCtxPopCurrent` / `cuCtxSetCurrent` |
| 記憶體歸屬 | 屬於 primary context | 屬於這個新 context（`cuPointerGetAttribute` 可查） |
| 銷毀後果 | 不會被你手滑銷毀 | **context 一死，裡面的記憶體全變野指標** |

### 一個重要的實務提醒

如果你想混用 Driver API 和 Runtime API（或任何用 Runtime API 寫的函式庫，例如 cuBLAS），**應該用 `cuDevicePrimaryCtxRetain`，不要用 `cuCtxCreate`**。

理由有三個，都在上面驗證過：

1. **生命週期安全**（證明 7）：primary context 由驅動參考計數管理，不會被誰手滑 `cuCtxDestroy` 掉。你自己建的 context 一死，裡面的記憶體全變野指標。
2. **沒有額外的 context 切換開銷**：GPU 在不同 context 之間切換是有實際成本的。
3. **語意單純**：所有 host thread 自動共用同一個 context，不必操心 stack 是 thread-local 這件事。

`cuCtxCreate` 建的是隔離的 context，只有在你**刻意要隔離**時才適合。

### 想自己驗證？改這幾個地方

[03_context_compare.cu](experiments/03_context_compare.cu) 只有一百多行，建議動手改。下面前兩項我已經實際跑過確認：

| 改什麼 | 怎麼改 | 會看到 |
|---|---|---|
| 證明 primary context 是 singleton | 呼叫兩次 `cuDevicePrimaryCtxRetain`，印出兩個 handle | **位址相同**（已驗證） |
| 證明 `cuCtxCreate` 不是 | 呼叫兩次 `cuCtxCreate`，印出兩個 handle | **位址不同**，每次都建新的（已驗證） |
| 證明 stack 是 thread-local | 開一個 `std::thread`，在裡面 `cuCtxGetCurrent` | 拿到 `NULL` — 主執行緒 push 的 context 它看不到（已驗證） |
| 加嚴證明 (6) | 在 primary 裡啟動 kernel 去寫 `p`，再 memcpy 回主機檢查 | 資料真的寫進去了 — 跨 context 連 kernel 存取都成功（已驗證） |

這是很多人從 Driver API 起手時會踩的坑：明明分配了記憶體，傳給 cuBLAS 卻說指標無效——因為根本不在同一個 context 裡。

---

# 三、nvcc 的編譯流程、目標碼類型與多架構設定

> 原題：請描述 nvcc 編譯器在編譯 CUDA 程式碼時的主要工作流程，包括生成目標碼的類型及其作用。如果需要支援多種 GPU 架構的二進制檔案，應該如何設置 nvcc 的編譯選項？

## 一句話版本

> **nvcc 是個「分流器」**：把 `.cu` 拆成 host 與 device 兩半，host 那半原封不動交給 gcc/MSVC，device 那半走自己的管線編成 GPU 碼，最後把 GPU 碼當成**一個位元組陣列嵌回 host 的執行檔裡**。

最後那句是整題的關鍵。你的 `a.out` 裡面**藏著一份 GPU 二進位**，不是另外一個檔案。

## 3-1 編譯流程

`nvcc --dryrun` 會印出它實際要執行的每一道指令，不用猜。[04_nvcc_pipeline.sh](experiments/04_nvcc_pipeline.sh) 幫你整理過：

```
gcc          -E          # 前處理（host 用）
cudafe++                 # 把 .cu 拆成「host 端 C++」與「device 端」
gcc          -E          # 前處理（device 用，這次帶 __CUDA_ARCH__）
cicc         -arch       # device 編譯器：device 碼 -> PTX
ptxas        -arch=sm_89 # PTX 組譯器：PTX -> cubin（內含 SASS）
fatbinary                # 把 cubin + PTX 打包成 fatbin
gcc          -c          # 編譯 host 端（fatbin 以位元組陣列嵌入）
nvlink                   # device 端連結
fatbinary    -link
gcc          -c
g++                      # 最後連結成執行檔
```

畫成圖比較好懂：

```
              your_code.cu
                   │
              ┌────┴────┐  cudafe++ 拆兩半
              ▼         ▼
        host C++    device 碼
              │         │
              │         ▼  cicc（device 編譯器）
              │       PTX  ← 虛擬指令集（文字）
              │         │
              │         ▼  ptxas（組譯器）
              │      cubin  ← SASS 真實機器碼
              │         │
              │         ▼  fatbinary
              │      fatbin  ← 打包多個架構的版本
              │         │
              ▼         ▼
        gcc -c ◄────嵌入為位元組陣列
              │
              ▼  g++ 連結
          執行檔（host 碼 + 內嵌 fatbin）
```

給 C/C++ 開發者的對照：

| nvcc 的階段 | 你熟悉的對應物 |
|---|---|
| `cudafe++` | 沒有對應物，這是 CUDA 特有的「拆檔」步驟 |
| `cicc` | 前端 + 最佳化器，像 `clang -emit-llvm` |
| PTX | **LLVM IR / Java bytecode** — 中間表示，文字格式 |
| `ptxas` | 組譯器，像 `as` |
| cubin | **`.o` 檔**（ELF 格式，真的是 ELF） |
| fatbin | **Apple 的 universal binary** — 一個檔多種架構 |
| `nvlink` | device 端的 `ld` |

## 3-2 目標碼有哪兩種，各自的作用

這是本題的核心。**只有兩種**，但功能完全不同：

| | PTX | SASS |
|---|---|---|
| 全名 | Parallel Thread eXecution | Streaming ASSembler |
| 性質 | **虛擬**指令集 | **真實**機器碼 |
| 格式 | 文字（可讀） | 二進位（裝在 cubin 裡） |
| 綁架構嗎 | **不綁**，向前相容 | **綁死**特定 compute capability |
| 誰產生 | `cicc` | `ptxas` |
| 執行時 | 需驅動 JIT 編譯成 SASS | 直接執行 |
| 作用 | **未來新卡的保險** | **當下的效能** |
| 對照 | LLVM IR | x86 機器碼 |

同一個 kernel 的兩種形式（實際 dump 出來的）：

```
PTX（虛擬暫存器 %r，跨架構）        SASS（真實暫存器 R，sm_89 專用）
  mov.u32     %r3, %ctaid.x;         S2R   R6, SR_CTAID.X ;
  mov.u32     %r4, %ntid.x;          S2R   R3, SR_TID.X ;
  mad.lo.s32  %r1, %r3, %r4, %r5;    IMAD  R6, R6, c[0x0][0x0], R3 ;
  setp.ge.s32 %p1, %r1, %r2;         ISETP.GE.AND P0, PT, R6, ... ;
```

`mad.lo.s32` → `IMAD`，`%r1` → `R6`。**PTX 是願望，SASS 是現實。**

### 還有兩個名詞要分清楚

| 名詞 | 是什麼 |
|---|---|
| **cubin** | 裝著某**一個**架構的 SASS 的檔案。是真正的 ELF 格式 |
| **fatbin** | 一個容器，裡面可以放**多個** cubin ＋ PTX。就是「胖」的意思 |

實測 `-arch=sm_89` 編出來的執行檔裡有什麼：

```
$ cuobjdump -lelf a.out          # 列出 cubin
ELF file    1: a.out.1.sm_89.cubin
ELF file    2: a.out.2.sm_89.cubin

$ cuobjdump -lptx a.out          # 列出 PTX
PTX file    1: a.out.1.sm_89.ptx
```

**兩個發現：**

1. 只寫 `-arch=sm_89`，nvcc **同時**放了 SASS 和 PTX。因為 `-arch=sm_XX` 是 `code=sm_XX,compute_XX` 的簡寫。
2. 每個架構出現**兩個** cubin 是正常的：一個來自主編譯，一個來自 device link 階段。

## 3-3 支援多種 GPU 架構

### 語法：`-gencode`

```bash
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_89,code=compute_89 \
     -o app app.cu
```

**兩個參數不一樣，這是最容易搞混的地方：**

| 參數 | 意思 | 對照 |
|---|---|---|
| `arch=compute_XX` | 用哪個**虛擬**架構編 PTX（前端） | `-std=c++17`：語言層級 |
| `code=sm_XX` | 產生哪個**真實**架構的 SASS | `-march=znver4`：目標機器 |
| `code=compute_XX` | **保留 PTX 本身**（不編成 SASS） | 保留 IR 給未來 JIT |

最後那行 `code=compute_89` 是關鍵，下面會證明為什麼。

### 實測：fatbin 裡真的有三種架構

```
$ cuobjdump -lelf b_multi
ELF file    1: b_multi.1.sm_75.cubin      ← Turing
ELF file    2: b_multi.2.sm_86.cubin      ← Ampere
ELF file    3: b_multi.3.sm_89.cubin      ← Ada
ELF file    4: b_multi.4.sm_75.cubin      （device link 的那一份）
ELF file    5: b_multi.5.sm_86.cubin
ELF file    6: b_multi.6.sm_89.cubin

$ cuobjdump -lptx b_multi
PTX file    1: b_multi.1.sm_89.ptx        ← 只留最高的當保險
```

### 體積代價（實測）

看 `.nv_fatbin` 區段（避開被靜態連結的 cudart 蓋掉）：

| 版本 | `.nv_fatbin` 大小 |
|---|---|
| 單一架構 `-arch=sm_89` | 4,632 bytes |
| 三種架構 + PTX | 12,376 bytes |

約 2.7 倍。**但這只是 fatbin 區段** — 整個執行檔因為靜態連結 cudart 有 ~1 MB，所以實際檔案只從 998 KB 變成 1006 KB，成長不到 1%。

編譯**時間**才是真代價：每個 `-gencode` 都要跑一次完整的 `cicc` + `ptxas`。

### 為什麼一定要留 PTX？兩個實測

**實驗 A：只留 PTX（`code=compute_75`），在 sm_89 上跑**

```bash
$ nvcc -gencode arch=compute_75,code=compute_75 -o d_ptxonly k.cu
$ cuobjdump -lelf d_ptxonly
（沒有任何 cubin）
$ ./d_ptxonly
kernel result: no error          ← 成功！
```

**成功了。** 驅動把 compute_75 的 PTX 即時編譯（JIT）成 sm_89 的 SASS。這就是**向前相容**：程式是 Turing 時代編的，卻能在 Ada 上跑。

JIT 結果會快取在 `~/.nv/ComputeCache`（可用 `CUDA_CACHE_DISABLE=1` 關閉）。

> 附帶測量：對這種小 kernel，JIT 的成本被 CUDA 初始化（~0.15 秒）完全蓋住，量不出差異。要在有大量或大型 kernel 的專案才會感覺到啟動變慢。

**實驗 B：只留舊架構的 SASS、不留 PTX**

```bash
$ nvcc -gencode arch=compute_75,code=sm_75 -o e_sassonly k.cu
$ cuobjdump -lptx e_sassonly
（沒有 PTX）
$ ./e_sassonly
kernel result: no kernel image is available for execution on the device   ← 失敗
```

**這就是最常見的部署錯誤。** `no kernel image is available for execution on the device` 幾乎都是這個原因：使用者的卡不在你的 `code=sm_XX` 清單裡，而你又沒留 PTX 當退路。

C/C++ 的比喻：實驗 B 像是你只發布了 x86 的執行檔給 ARM 使用者。實驗 A 像是你發布了原始碼，對方自己編。

### 實務建議

**開發時**（只跑自己的卡）— 編譯最快：

```bash
nvcc -arch=sm_89 -O2 -o app app.cu       # 或用 -arch=native（CUDA 11.5+）
```

**發布時** — 涵蓋常見架構，並留 PTX：

```bash
nvcc -O2 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  -o app app.cu
```

**規則就一條：清單裡最高那個架構，一定要多寫一行 `code=compute_XX` 留 PTX。**

CMake 的等價寫法：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)   # 自動幫你補 PTX
```

### 好用的檢查指令

| 指令 | 用途 |
|---|---|
| `nvcc --dryrun -o x x.cu` | 看實際執行的每個編譯階段 |
| `cuobjdump -lelf <bin>` | 列出裡面有哪些架構的 cubin |
| `cuobjdump -lptx <bin>` | 列出裡面有哪些 PTX |
| `cuobjdump -sass <bin>` | 反組譯出 SASS |
| `cuobjdump -ptx <bin>` | 抽出 PTX 原文 |
| `nvcc -ptx x.cu` | 只編到 PTX 就停 |
| `nvcc -cubin x.cu` | 只編到 cubin 就停 |
| `nvcc --list-gpu-arch` | 這版 nvcc 支援哪些架構 |

**部署前的自我檢查：`cuobjdump -lelf` 加 `-lptx` 看一眼**，比使用者回報「跑不起來」便宜太多。

---

# 四、P-State

> 原題：請說明 P-State 的定義和作用，GPU 在運行過程中如何根據負載動態調整 P-State？如果強制 GPU 始終運行在 P0 狀態，可能會有哪些影響？

## 4-1 定義與作用

**P-State（Performance State，效能狀態）是一組預先定義好的「電壓 + 各個時脈域」設定的組合。**

- 編號 **P0 到 P12**
- **P0 = 最高效能**，P12 = 最省電
- 數字越大越省電（跟直覺相反，記一下）

最重要的一個觀念，也是這題的陷阱：

> **P-State 不是「時脈」，而是一整組 clock domain 的設定。**

一個 GPU 有好幾個獨立的時脈域：SM 時脈、記憶體時脈、視訊編解碼時脈…… P-State 一次調整一整組，不是只調一個數字。

C/C++ 對照：這跟 CPU 的 P-state（`cpufreq` governor、Intel SpeedStep）是同一個概念，都源自 ACPI 規範。差別是 GPU 的記憶體時脈變化幅度大得多。

**作用**：省電與散熱。GPU 閒置時（例如只在跑桌面合成），把記憶體時脈從 8 GHz 降到 810 MHz 可以省下大量功耗，本機實測是 45 W → 17 W。

## 4-2 如何根據負載動態調整（實測）

[06_pstate_monitor.sh](experiments/06_pstate_monitor.sh) 每 0.5 秒取樣一次，在負載進來的瞬間觀察。先看硬體天花板：

```
GPU                              clocks.max.sm  clocks.max.mem  power.default_limit
NVIDIA GeForce RTX 4060 Laptop   3105 MHz       8001 MHz        115.00 W
```

實測輸出（節錄）：

```
t        pstate  util%  sm_MHz    mem_MHz   power_W  temp
0        P5      0      2475      810       18.31    63     ← 閒置
.5       P0      100    1950      8001      26.43    72     ← 負載進來，0.5 秒內就跳 P0
1.0      P0      100    1770      8001      44.33    73
2.5      P0      100    1920      8001      44.98    74
3.5      P3      100    1980      7001      45.02    76     ← 開始震盪降階
5.0      P3      100    1965      7001      44.94    76
8.5      P0      100    1950      8001      44.98    72
11.5     P0      100    1920      8001      44.99    72
--- 負載結束 ---
+2s      P4      0      2475      6001      21.02    60
+4s      P5      0      2475      810       17.74    58     ← 約 4 秒回到 P5
```

### 四個從數據看出來的結論

**(1) 反應非常快**

負載進來 **0.5 秒內**就從 P5 跳到 P0；負載結束約 **4 秒**回到 P5。這是驅動與硬體自動做的，程式不需要（也無法直接）介入。

**(2) P0 的 SM 時脈竟然比 P5 低 — 這題最反直覺的地方**

| | SM 時脈 | 記憶體時脈 | 功耗 | 溫度 |
|---|---|---|---|---|
| 閒置 P5 | **2475 MHz** | 810 MHz | 17.6 W | 63°C |
| 負載 P0 | **~1950 MHz** | 8001 MHz | 45 W | 74°C |
| 硬體上限 | 3105 MHz | 8001 MHz | 115 W | — |

**P0 的 SM 時脈反而低了 500 MHz。** 而且兩個狀態都**沒有**達到 3105 MHz 的上限。

原因由程式抓到的證據直接說明：

```
=== 負載中的時脈受限原因 ===
      Performance State     : P0
          SW Power Cap      : Active        ← 這就是答案
          HW Slowdown       : Not Active
          SW Thermal Slowdown : Not Active
```

**`SW Power Cap : Active`。** 解釋：

- 閒置時沒有實際工作，高時脈幾乎不耗電，所以讀數接近 boost 值
- 一旦真的有工作，功耗限制立刻生效，時脈被壓下來
- 所以 **P0 的意思是「允許使用最高效能狀態」，不是「保證跑在最高時脈」**

實際時脈是 boost 演算法在功耗與溫度預算內動態決定的。這個區別是回答第三小問的關鍵。

**(3) 真正被 P-State 拉上去的是記憶體時脈**

810 → 8001 MHz，**約 10 倍**。這才是 P5 → P0 的主要內容。再次印證「P-State 是一整組設定」。

**(4) 持續負載下狀態會震盪，不是跳到 P0 就固定**

看數據裡的 P0 / P3 交替，記憶體時脈跟著 8001 / 7001 / 6001 降階。GPU 一直在重新協商功耗與溫度預算。

> 這也是我在做這個實驗時的一個修正：我一開始看到單次取樣是 P4 就以為「記憶體吃重的負載會停在 P4」，多跑幾次才發現那是震盪中的一個取樣點。**單次 `nvidia-smi` 讀數不能代表穩態**，要連續取樣。

## 4-3 強制 GPU 始終運行在 P0 的影響

### 怎麼做（需要 root）

嚴格來說沒有「直接設定 P-State」的指令，實務上是用鎖時脈與 persistence mode 達到等效：

```bash
sudo nvidia-smi -pm 1                       # persistence mode：驅動常駐
sudo nvidia-smi -lgc 2100,2100              # 鎖定 SM 時脈
sudo nvidia-smi --lock-memory-clocks=8001   # 鎖定記憶體時脈
sudo nvidia-smi -rgc                        # 解除 SM 時脈鎖
sudo nvidia-smi -rmc                        # 解除記憶體時脈鎖
```

> 本機沒有 root，所以我**沒有實際套用**這些設定（也不該擅自改動系統狀態）。以下影響分析，是建立在上面已經實測到的機制之上。

### 好處

| 好處 | 為什麼 |
|---|---|
| **消除 P-State 轉換延遲** | 實測顯示轉換要 0.5 秒。短小而頻繁的 kernel 可能整段都跑在低 P-State 上就結束了 |
| **效能可重現** | 做 benchmark 時最重要的一點。不鎖時脈，同樣的程式跑兩次差幾個百分比是常態 |
| **避免降頻造成的抖動** | 實測看到的 P0/P3 震盪會讓 latency 不穩 |

### 代價

| 代價 | 實測依據 |
|---|---|
| **功耗大幅上升** | 本機閒置 17.6 W vs 負載 45 W。鎖 P0 就是閒置也要付 45 W 這一級的代價 |
| **溫度上升** | 63°C → 74~76°C。筆電尤其明顯（散熱與 CPU 共用） |
| **可能反而更慢** | 這是最反直覺的一點，見下面說明 |
| 風扇噪音、電池續航 | 筆電上很有感 |
| 硬體長期壽命 | 持續高溫對元件不利 |

### 為什麼「鎖 P0 可能反而更慢」

這是我從實測數據推出來的，值得單獨講：

本機負載中 **`SW Power Cap : Active`**，功耗預算是硬限制。如果你把 SM 時脈鎖在一個高值：

1. 功耗預算被 SM 吃掉更多
2. 記憶體子系統能分到的就變少
3. 而**大多數 CUDA 程式是 memory-bound**（瓶頸在搬資料，不在算）
4. 結果：算得更快、但資料餵不上來 → **整體變慢**

實測的旁證：記憶體吃重的負載，SM 時脈自動降到 ~1710 MHz（比 compute-bound 的 ~1950 MHz 更低）。**硬體自己在做功耗預算的分配，而且它通常分配得比手動鎖定更好。**

另外，鎖了時脈之後如果撞到**溫度**牆，硬體會觸發 `HW Thermal Slowdown`，那個降頻幅度比 P-State 機制粗暴得多。

### 實務建議

| 情境 | 建議 |
|---|---|
| **跑 benchmark、要可重現的數字** | 鎖時脈。這是唯一真正需要的場合 |
| 資料中心長時間高負載訓練 | 開 persistence mode，但時脈交給硬體管 |
| 一般開發、桌機/筆電日常使用 | **不要鎖**。硬體的動態調整通常比你聰明 |
| 覺得 kernel 啟動有延遲 | 先開 persistence mode（`-pm 1`），成本低很多 |

**Persistence mode 和鎖時脈是兩件事**，常被混為一談：

- `-pm 1` 只是讓驅動常駐，避免每次程式啟動都重新初始化 GPU（省的是 CUDA 初始化那 ~0.15 秒），成本很低
- `-lgc` 才是鎖時脈，代價才是上面那張表

本機 persistence mode 實測已經是 `Enabled`。

---

# 五、用 nvidia-smi 查詢即時狀態

> 原題：請說明如何使用 nvidia-smi 工具查詢 GPU 的即時狀態訊息，包括 GPU 的利用率、功耗和溫度等數據。舉例說明如何使用該工具輸出 GPU0 的即時功耗和記憶體的使用情況。

## 5-1 三種用法，選對的那個

| 用法 | 適合 | 缺點 |
|---|---|---|
| `nvidia-smi` | 人眼快速看一下 | 欄位固定，不好用程式解析 |
| `nvidia-smi --query-gpu=... --format=csv` | **寫腳本、記錄數據** | 要自己挑欄位 |
| `nvidia-smi dmon` | 長時間看趨勢 | 欄位是預設群組，不能任選 |

實務上第二種用得最多。

## 5-2 本題要的答案：輸出 GPU0 的即時功耗與記憶體使用

**最直接的一行：**

```bash
nvidia-smi -i 0 --query-gpu=power.draw,memory.used,memory.total,memory.free --format=csv
```

實際輸出（負載中，`05_gpu_load` 配置了 2 GB）：

```
power.draw [W], memory.used [MiB], memory.total [MiB], memory.free [MiB]
44.94 W, 2163 MiB, 8188 MiB, 6025 MiB
```

**三個參數的意思：**

| 參數 | 作用 |
|---|---|
| `-i 0` | 只看 GPU 0（多卡機器很重要） |
| `--query-gpu=a,b,c` | 指定欄位，**逗號分隔、中間不能有空格** |
| `--format=csv` | 輸出格式（必填） |

### 變化寫法

**去掉標題與單位，方便腳本處理：**

```bash
$ nvidia-smi -i 0 --query-gpu=power.draw,memory.used --format=csv,noheader,nounits
44.94, 2163
```

在 bash 裡取單一數值：

```bash
w=$(nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits)
echo "目前功耗 = $w W"
```

**每秒持續取樣（`-l 1`）：**

```bash
$ nvidia-smi -i 0 --query-gpu=timestamp,power.draw,memory.used,utilization.gpu --format=csv -l 1
timestamp, power.draw [W], memory.used [MiB], utilization.gpu [%]
2026/08/17 22:35:54.844, 44.94 W, 2163 MiB, 100 %
2026/08/17 22:35:55.848, 45.06 W, 2163 MiB, 100 %
2026/08/17 22:35:56.848, 45.05 W, 2163 MiB, 100 %
```

**存成 CSV 檔記錄一整段時間：**

```bash
nvidia-smi -i 0 --query-gpu=timestamp,power.draw,memory.used,utilization.gpu,temperature.gpu \
           --format=csv -l 1 > gpu_log.csv
```

## 5-3 一次看完整狀態（實務最常用的一行）

```bash
nvidia-smi -i 0 --query-gpu=pstate,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,power.default_limit,memory.used,memory.total,temperature.gpu --format=csv
```

```
pstate, utilization.gpu [%], utilization.memory [%], clocks.current.sm [MHz],
clocks.current.memory [MHz], power.draw [W], power.default_limit [W],
memory.used [MiB], memory.total [MiB], temperature.gpu

P5, 0 %, 0 %, 2475 MHz, 810 MHz, 17.33 W, 115.00 W, 13 MiB, 8188 MiB, 59
```

這一行就是第四題那些數據的來源。

## 5-4 dmon：緊湊的滾動監控

```bash
$ nvidia-smi dmon -s puct -c 4
# gpu   pwr  gtemp  mtemp    sm   mem   enc   dec   jpg   ofa   mclk   pclk  rxpci  txpci
# Idx     W      C      C     %     %     %     %     %     %    MHz    MHz   MB/s   MB/s
    0    44     73      -   100    13     0     0     0     0   7001   1950      0      0
    0    44     73      -   100    13     0     0     0     0   7001   1785      0      0
    0    44     73      -   100    12     0     0     0     0   7001   1800      0      0
```

`-s` 選欄位群組，可以組合：

| 字母 | 內容 |
|---|---|
| `p` | 功耗與溫度 |
| `u` | 使用率（sm / mem / 編解碼器） |
| `c` | 時脈（mclk 記憶體 / pclk 處理器） |
| `t` | PCIe 收發流量 |
| `m` | 記憶體使用 |

`-c 4` 是取 4 筆就結束（不加會一直跑）。

## 5-5 查哪個 process 吃掉顯存

`memory.used` 只給總量，要看是誰佔的：

```bash
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
624458, ./05_gpu_load, 2142 MiB
```

顯存爆掉時第一個要跑的就是這條。

## 5-6 常用欄位速查

| 類別 | 欄位 | 說明 |
|---|---|---|
| **功耗** | `power.draw` | 目前功耗（W） |
| | `power.default_limit` | 預設功耗上限（本機 115 W） |
| | `power.max_limit` | 硬體最大上限（本機 140 W） |
| **記憶體** | `memory.used` / `free` / `total` | 顯存使用／剩餘／總量（MiB） |
| | `utilization.memory` | 記憶體**頻寬**使用率，不是佔用率 |
| **運算** | `utilization.gpu` | 有 kernel 在跑的**時間**佔比 |
| **時脈** | `clocks.sm` / `clocks.mem` | 目前時脈 |
| | `clocks.max.sm` / `clocks.max.mem` | 硬體上限 |
| **狀態** | `pstate` | P0（最高效能）~ P12（最省電） |
| | `persistence_mode` | Persistence mode 開了沒 |
| **溫度** | `temperature.gpu` | 核心溫度（°C） |
| **識別** | `name` / `compute_cap` / `driver_version` | 卡名／計算能力／驅動版本 |

完整清單：`nvidia-smi --help-query-gpu`（本機約 218 個欄位）。

## 5-7 兩個一定會誤解的欄位

這兩個坑幾乎每個人都踩過：

**`utilization.gpu` 不是「算力使用率」**

它的定義是「**這段取樣時間內，有多少比例的時間至少有一個 kernel 在跑**」。

所以：

```c
kernel<<<1, 1>>>(...);   // 只用 1 個 thread，24 個 SM 裡 23.97 個在閒置
```

這樣也可能顯示 **100%**。看到 100% 只代表「GPU 沒閒著」，**完全不代表跑得有效率**。要看真實效率得用 Nsight Compute。

**`utilization.memory` 不是 `memory.used`**

| 欄位 | 意思 |
|---|---|
| `memory.used` | 佔了多少 **MiB** 顯存（容量） |
| `utilization.memory` | 記憶體**頻寬**有多忙（百分比） |

上面 dmon 的實測就是好例子：`mem` 欄位只有 12~13%，但那個 kernel 佔了 2142 MiB 顯存。**容量吃很多，頻寬很閒**（因為那是 compute-bound 的 kernel）。

## 5-8 版本欄位的老陷阱

```bash
$ nvidia-smi | head -3
| NVIDIA-SMI 580.167.08   Driver Version: 580.167.08   CUDA Version: 13.0 |
```

這裡的 **`CUDA Version: 13.0` 是「驅動最高支援到的版本」，不是你裝的 toolkit 版本。**

本機真正的 toolkit 是 12.4：

```bash
$ nvcc --version | tail -2
Cuda compilation tools, release 12.4, V12.4.131
```

**兩者獨立，看 toolkit 版本要用 `nvcc --version`。** 這是新手最常見的困惑。

---

# 附錄：CUDA 術語表

以「C/C++ 開發者第一次看到會卡住」為標準挑選。有對應概念的都給了對照。

## A. 執行模型

| 術語 | 中文 | 解釋 | C/C++ 對照 |
|---|---|---|---|
| **Host** | 主機 | CPU 與它的記憶體（`malloc` 的那個世界） | 你原本熟悉的一切 |
| **Device** | 裝置 | GPU 與它的顯示記憶體 | 一台獨立的協同處理器 |
| **Kernel** | 核心函式 | 標了 `__global__`、在 GPU 上跑的函式。**注意**：跟「作業系統核心」毫無關係 | 一個會被幾百萬個 lane 同時執行的函式 |
| **Thread** | 執行緒 | 最小執行單位。**極輕量**，不是 OS thread | 一個 SIMD lane |
| **Warp** | — | **32 個 thread 綁成一組**，硬體排程的最小單位。無中文通譯，直接講 warp | 一個 AVX 向量（但寬度 32） |
| **Lane** | 通道 | warp 裡的一個位置（0~31） | SIMD 向量裡的一個格子 |
| **Block / CTA** | 執行緒區塊 | 一群 thread（本機最多 1024 個），**整組被丟到同一個 SM**，可以共用 shared memory、可以互相同步 | 「綁在同一顆核心上的一組工作」 |
| **Grid** | 網格 | 一次 kernel 啟動的所有 block | `parallel_for` 的整個迭代範圍 |
| **SIMT** | — | Single Instruction, Multiple Threads。NVIDIA 的執行模型 | **SIMD + 每個 lane 可獨立遮罩** |

### 內建變數（在 kernel 裡直接可用，不用宣告）

| 變數 | 意思 |
|---|---|
| `threadIdx.x/.y/.z` | 我在 block 裡是第幾個 thread |
| `blockIdx.x/.y/.z` | 我的 block 在 grid 裡是第幾個 |
| `blockDim.x/.y/.z` | 一個 block 有幾個 thread |
| `gridDim.x/.y/.z` | 一個 grid 有幾個 block |

幾乎每個 kernel 的第一行都是這個慣用句：

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;   // 算出「我負責第幾筆資料」
```

## B. 硬體

| 術語 | 中文 | 解釋 | CPU 對照 |
|---|---|---|---|
| **SM** | 串流多處理器 | GPU 的「核心」。本機有 24 個 | CPU core |
| **CUDA Core** | — | SM 內的運算單元。**行銷術語**，不要當成 CPU core 理解 | ALU |
| **Tensor Core** | 張量核心 | 專做矩陣乘加的單元，AI 用 | Intel AMX |
| **Warp Scheduler** | Warp 排程器 | 每 cycle 挑一個 ready 的 warp 發射指令 | 指令排程器 |
| **Register File** | 暫存器檔案 | 本機每 SM 65536 個 32-bit 暫存器 | 暫存器（但大好幾個數量級） |
| **SFU** | 特殊函式單元 | 硬體算 `sin`/`cos`/`exp`/`rsqrt` | 無，CPU 用軟體算 |
| **Compute Capability** | 計算能力 | GPU 的架構版本，本機 8.9 | `-march=` 的目標架構 |
| **Ada Lovelace** | — | 本機 GPU 的架構代號（RTX 40 系列） | 如 Zen 4、Golden Cove |

## C. 記憶體階層

**這是 CUDA 效能的另一半，C/C++ 沒有對應概念，值得記熟。**

| 類型 | 範圍 | 速度 | 大小（本機） | 說明 |
|---|---|---|---|---|
| **Register** | 單一 thread | 最快 | 每 SM 64K 個 | 區域變數預設放這 |
| **Shared Memory** | 同一 block | 很快 | 每 SM 100 KB | **可程式控制的 L1**，用 `__shared__` 宣告 |
| **L1 / L2 Cache** | SM / 全域 | 快 | — | 自動，跟 CPU 一樣 |
| **Global Memory** | 全部 thread | **慢**（400~800 cycle） | 8 GB | `cudaMalloc` 拿到的就是這個 |
| **Constant Memory** | 全部（唯讀） | 快（有專用 cache） | 64 KB | `__constant__` |
| **Local Memory** | 單一 thread | **慢** | — | **名字騙人**：其實在 global memory 上。暫存器不夠用時編譯器把變數「溢出」到這裡，是效能殺手 |

| 術語 | 解釋 |
|---|---|
| **Coalescing（合併存取）** | 同一 warp 的 32 個 thread 存取**連續**位址時，硬體會合併成少數幾次交易。**這是 CUDA 最重要的效能規則之一**，比 warp 滿不滿更常影響效能 |
| **Bank Conflict** | Shared memory 分成 32 個 bank，同 warp 多個 thread 打到同一 bank 會被序列化 |
| **Pinned Memory（page-locked）** | Host 端不會被 OS 換出的記憶體（`cudaMallocHost`），傳輸快很多 |
| **Unified Memory** | `cudaMallocManaged`，CPU/GPU 共用一個指標，驅動自動搬資料。方便但通常較慢 |

## D. 函式修飾字與語法

| 語法 | 意思 |
|---|---|
| `__global__` | **kernel**。從 host 呼叫、在 device 執行。回傳型別必須是 `void` |
| `__device__` | 只能從 device 呼叫、在 device 執行（GPU 上的一般函式） |
| `__host__` | 一般 CPU 函式（預設值，通常不用寫） |
| `__host__ __device__` | 兩邊都編一份，兩邊都能呼叫 |
| `__shared__` | 宣告 shared memory 變數 |
| `__constant__` | 宣告 constant memory 變數 |
| `<<<grid, block>>>` | **kernel 啟動語法**。C++ 沒有這個語法，是 nvcc 的擴充 |
| `__syncthreads()` | **block 內**的屏障同步。像 `pthread_barrier_wait`，但只在 block 內、且非常便宜 |
| `__syncwarp()` | warp 內的同步（Volta 之後有時需要） |

**`__syncthreads()` 的地雷**：所有 thread 必須**都**到得了它。放在 divergent 的 `if` 裡面是未定義行為，會 hang 或給錯誤結果。

```c
if (tid < 10) { __syncthreads(); }   // ❌ 錯！有些 thread 到不了
if (tid < 10) { ... }
__syncthreads();                     // ✅ 對
```

## E. 編譯與工具鏈

| 術語 | 解釋 | C/C++ 對照 |
|---|---|---|
| **nvcc** | CUDA 編譯器。把 device 碼抽出來自己編，host 碼丟給 gcc/MSVC | 一個會分流的 wrapper compiler |
| **PTX** | 虛擬指令集，**文字格式**，跨架構、可 JIT | LLVM IR / Java bytecode |
| **SASS** | 真正的機器碼，綁死特定架構 | x86 machine code |
| **cubin** | 裝著**單一**架構 SASS 的檔案，實際是 ELF 格式 | `.o` 檔 |
| **fatbin** | 容器，可裝**多個** cubin ＋ PTX | Apple 的 universal binary |
| **JIT** | 執行期把 PTX 編成 SASS，驅動自動做 | JIT 編譯 |
| **`-arch=sm_89`** | 指定目標架構（本機 8.9）。是 `code=sm_89,compute_89` 的簡寫 | `-march=native` |
| **Toolkit** | nvcc + 函式庫 + 標頭檔，本機 12.4 | SDK |
| **Driver** | 顯示卡驅動，本機 580.167.08 | 核心模組 |

### nvcc 的編譯階段（`--dryrun` 可看到）

| 階段工具 | 做什麼 | C/C++ 對照 |
|---|---|---|
| **cudafe++** | 把 `.cu` 拆成 host C++ 與 device 兩半 | 無對應物，CUDA 特有 |
| **cicc** | device 編譯器：device 碼 → PTX | `clang -emit-llvm` |
| **ptxas** | PTX 組譯器：PTX → cubin | `as` |
| **fatbinary** | 把 cubin + PTX 打包成 fatbin | `lipo` |
| **nvlink** | device 端連結器 | `ld` |

### `-gencode` 的兩個參數（最容易搞混）

| 寫法 | 意思 | C/C++ 對照 |
|---|---|---|
| `arch=compute_XX` | 用哪個**虛擬**架構編 PTX | `-std=c++17`（語言層級） |
| `code=sm_XX` | 產生哪個**真實**架構的 SASS | `-march=znver4`（目標機器） |
| `code=compute_XX` | **保留 PTX 本身**，給未來的卡 JIT 用 | 保留 IR |

### 檢查工具

| 指令 | 用途 |
|---|---|
| `cuobjdump -lelf <bin>` | 列出裡面有哪些架構的 cubin |
| `cuobjdump -lptx <bin>` | 列出裡面有哪些 PTX |
| `cuobjdump -sass <bin>` | 反組譯出 SASS |
| `nvcc --dryrun` | 看實際執行的每個編譯階段 |
| `nvcc --list-gpu-arch` | 這版 nvcc 支援哪些架構 |

**Toolkit 版本 ≠ Driver 版本，兩者獨立。** `nvidia-smi` 顯示的 CUDA 版本是「驅動最高支援到」，不是「你裝的 toolkit」。這是新手最常見的困惑之一。

## E2. 電源、時脈與監控

| 術語 | 中文 | 解釋 | 對照 |
|---|---|---|---|
| **P-State** | 效能狀態 | 一組預定義的「電壓 + 各時脈域」組合。**P0 最高效能**，P12 最省電。源自 ACPI | CPU 的 SpeedStep / `cpufreq` |
| **Clock Domain** | 時脈域 | GPU 有多個獨立時脈：SM、記憶體、視訊等。P-State 一次調整一整組 | — |
| **SM Clock** | SM 時脈 | `clocks.sm`，運算單元的頻率 | CPU 核心頻率 |
| **Memory Clock** | 記憶體時脈 | `clocks.mem`。本機閒置 810 MHz、負載 8001 MHz，差 10 倍 | RAM 頻率 |
| **Boost Clock** | — | 在功耗與溫度預算內動態超上去的時脈 | Turbo Boost |
| **SW Power Cap** | 軟體功耗上限 | 驅動因功耗預算而降頻。**顯示 Active 就代表被功耗限制住了** | RAPL |
| **HW Slowdown** | 硬體降速 | 硬體保護機制觸發的強制降頻，比 P-State 粗暴得多 | 熱保護降頻 |
| **Thermal Throttling** | 溫度降頻 | 因溫度過高而降頻 | 同名 |
| **TGP / Power Limit** | 功耗上限 | 本機預設 115 W、硬體上限 140 W | TDP |
| **Persistence Mode** | 常駐模式 | 讓驅動常駐，避免每次程式啟動都重新初始化 GPU。**與鎖時脈是兩件事** | 常駐服務 |
| **Dynamic Boost** | — | 筆電上 GPU 與 CPU 共用功耗預算，動態分配 | — |

### 兩個一定會誤解的 nvidia-smi 欄位

| 欄位 | 常見誤解 | 真正的意思 |
|---|---|---|
| `utilization.gpu` | 算力使用率 | 「有多少比例的**時間**至少有一個 kernel 在跑」。用 1 個 thread 也能顯示 100% |
| `utilization.memory` | 顯存佔用率 | 記憶體**頻寬**忙碌程度。要看容量請用 `memory.used` |

### nvidia-smi 常用指令

| 指令 | 用途 |
|---|---|
| `nvidia-smi` | 人眼看的總表 |
| `nvidia-smi -i 0 --query-gpu=<欄位> --format=csv` | 挑欄位輸出，適合腳本 |
| `--format=csv,noheader,nounits` | 去掉標題與單位，好解析 |
| `-l 1` | 每秒重複取樣 |
| `nvidia-smi dmon -s puct` | 緊湊的滾動監控 |
| `nvidia-smi --query-compute-apps=...` | 看哪個 process 吃顯存 |
| `nvidia-smi -q -d PERFORMANCE` | 查降頻原因（Clocks Event Reasons） |
| `nvidia-smi --help-query-gpu` | 完整欄位清單（本機約 218 個） |
| `sudo nvidia-smi -pm 1` | 開 persistence mode（需 root） |
| `sudo nvidia-smi -lgc a,b` / `-rgc` | 鎖／解鎖 SM 時脈（需 root） |

## F. API 與資源管理

| 術語 | 解釋 | C/C++ 對照 |
|---|---|---|
| **Runtime API** | `cuda*` 系列，高階好用 | 靜態連結、直接呼叫 |
| **Driver API** | `cu*` 系列，低階囉唆 | `dlopen` + `dlsym` |
| **Context** | GPU 上的執行環境，有自己的位址空間 | **一個行程** |
| **Primary Context** | 每個裝置唯一、驅動管理的 context，Runtime API 用的就是它 | singleton |
| **Context Stack** | 每個 host thread 一個的 context 堆疊 | `thread_local` 的堆疊 |
| **Module** | 載入到 context 裡的一包 device 程式碼 | `.so` / DLL |
| **Stream** | 一個依序執行的命令佇列。不同 stream 可以並行 | 工作佇列 / `std::async` 的 pipeline |
| **Event** | 插在 stream 裡的時間戳／同步點 | condition variable + 計時器 |
| **`cudaDeviceSynchronize()`** | 等 GPU 上所有工作做完 | `thread.join()` |

**最容易踩的坑：kernel 啟動是非同步的。**

```c
kernel<<<g, b>>>(...);      // 立刻返回！GPU 可能還沒開始跑
printf("done\n");           // 這行會先印出來
cudaDeviceSynchronize();    // 到這裡才真的等它做完
```

而且因為非同步，**kernel 內部的錯誤不會從啟動那行回報**。抓錯要這樣寫：

```c
kernel<<<g, b>>>(...);
cudaError_t e = cudaGetLastError();        // 抓「啟動參數」的錯（例如 block 太大）
if (e) printf("launch: %s\n", cudaGetErrorString(e));
e = cudaDeviceSynchronize();               // 抓「執行期」的錯（例如非法記憶體存取）
if (e) printf("exec: %s\n", cudaGetErrorString(e));
```

## G. 效能術語

| 術語 | 中文 | 解釋 |
|---|---|---|
| **Occupancy** | 佔用率 | 駐留 warp 數 ÷ 硬體上限（本機 48）。太低就沒東西可以隱藏延遲 |
| **Latency Hiding** | 延遲隱藏 | 一個 warp 在等記憶體時換另一個跑。GPU 效能的根本機制 |
| **Warp Divergence** | 執行緒分歧 | 同 warp 內走不同分支 → 序列化執行，時間相加 |
| **Active Mask** | 活躍遮罩 | 32-bit 遮罩，標示 warp 裡哪些 lane 現在有效 |
| **Resident** | 駐留 | Block/warp 已配置在 SM 上、佔著資源 |
| **Issue Slot** | 發射槽 | 一個 cycle 裡發射一條指令的機會。**閒置的 lane 照樣佔用它**——這就是第一題的核心 |
| **Memory-bound** | 記憶體受限 | 瓶頸在搬資料，不在算。**大多數 CUDA 程式都是這種** |
| **Compute-bound** | 運算受限 | 瓶頸在算 |
| **Nsight Compute / Systems** | — | 官方 profiler。前者分析單一 kernel，後者看整體時間軸 |

## H. 常見誤解速查

| 誤解 | 真相 |
|---|---|
| CUDA thread 像 OS thread，很貴 | 極輕量，開幾百萬個是正常操作 |
| Kernel 是作業系統核心 | 完全無關，就是「在 GPU 上跑的函式」 |
| Local Memory 是快的本地記憶體 | **在 global memory 上，很慢**。出現代表暫存器溢出 |
| CUDA Core 像 CPU core | 只是一個 ALU，行銷術語 |
| `nvidia-smi` 顯示的 CUDA 版本＝我裝的 toolkit | 那是**驅動最高支援**的版本 |
| blockDim 一定要是 32 的倍數 | 更精準的說法：**不滿的那個 warp 佔全體比例要小**（見 1-3 結論 2） |
| kernel 啟動後就跑完了 | 非同步。要 `cudaDeviceSynchronize()` |
| kernel 裡的錯誤會從啟動那行回報 | 不會，要用 `cudaGetLastError()` + 同步後再查一次 |
| Driver API 比較快 | 效能一樣，差別只在控制粒度與部署彈性 |
| A context 的指標在 B context 完全不能用 | **過時說法**。有 UVA 時同卡跨 context 存取會成功（實測見證明 6）。真正的分界是**生命週期**：context 一死指標就變野的 |
| `utilization.gpu` 100% 代表 GPU 用滿了 | 只代表「有 kernel 在跑」。用 1 個 thread 也能顯示 100% |
| `utilization.memory` 是顯存佔用率 | 那是**頻寬**忙碌度。容量看 `memory.used` |
| `-arch=sm_89` 只放 SASS | 也會放 PTX（是 `code=sm_89,compute_89` 的簡寫），實測見 3-2 |
| P0 代表跑在最高時脈 | **不是**。P0 是「允許最高效能狀態」，實際時脈由功耗/溫度預算決定。本機 P0 的 SM 時脈反而比閒置的 P5 低 |
| P-State 就是時脈 | 是**一整組** clock domain 的設定。P5→P0 主要是記憶體時脈 810→8001 MHz |
| 鎖 P0 一定更快 | 可能更慢：功耗預算被 SM 吃掉，記憶體分到的變少，而多數 kernel 是 memory-bound |
| Persistence mode 等於鎖時脈 | 兩件事。`-pm 1` 只是驅動常駐，成本低；`-lgc` 才是鎖時脈 |
| `no kernel image is available` 是驅動壞了 | 幾乎都是編譯沒涵蓋該架構、又沒留 PTX（實測見 3-3 實驗 B） |
