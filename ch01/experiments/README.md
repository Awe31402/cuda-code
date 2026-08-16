# ch01 實測程式

[../ch01-questions.md](../ch01-questions.md) 裡所有數據的來源。三支程式都可以直接重跑驗證。

## 怎麼跑

```bash
cd ch01/experiments
make run          # 編譯並依序執行三支
```

或分開來：

```bash
make              # 只編譯
./01_device_query
./02_warp_blocksize
./03_context_compare
make clean
```

換不同 GPU 時，先跑 `./01_device_query` 看 compute capability，再指定：

```bash
make ARCH=sm_86   # 例如 RTX 30 系列
```

## 三支程式各在證明什麼

| 檔案 | 對應問題 | 證明的事 |
|---|---|---|
| [01_device_query.cu](01_device_query.cu) | 第 1 題 | 抓本機真實硬體規格：SM 數、warpSize、每 SM 上限。順便算出不同 blockDim 的 occupancy |
| [02_warp_blocksize.cu](02_warp_blocksize.cu) | 第 1-3 題 | warp 沒填滿 32 個 thread 會慢多少 |
| [03_context_compare.cu](03_context_compare.cu) | 第 2-3 題 | `cuCtxCreate` 建的 context 與 Runtime API 的 primary context 是**不同**的東西；並釐清「跨 context 指標不能用」這個常見誤解 |

## 02 的實驗設計（重點是控制變因）

- 總 thread 數固定 4,194,304，每個 thread 運算量完全相同
- **只改 blockDim**，grid 跟著調整，所以總工作量不變
- 因此時間差異只可能來自兩個原因：**warp 沒填滿** 和 **occupancy**
- 每組先 warmup 一次排除首次啟動成本，再取 20 次平均
- 用 `cudaEvent` 計時（GPU 端計時，不受 CPU 端非同步影響）

### 本機結果（RTX 4060 Laptop, sm_89）

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

三個結論：

1. **blockDim=1 慢 33.5 倍** — 跟理論值 32 幾乎吻合，證明硬體以 32 lane 為單位發射
2. **33 慢 1.89 倍，但 63 幾乎不慢** — 兩個都不是 32 的倍數。懲罰來自「不滿的那個 warp 佔全體比例」，不是「是否為 32 的倍數」
3. **32 就算 warp 全滿仍比 64 慢 7%** — 這是 occupancy 問題：blockDim=32 撞到「每 SM 最多 24 個 block」的上限，只有 50% occupancy

## 03 的重點：一個被推翻的常見說法

程式用 Driver API 當「觀察工具」，看 Runtime API 背後做了什麼。八個編號輸出各證明一件事：

| 輸出 | 證明 |
|---|---|
| `[1][2]` | context 是延遲建立的，`cuInit` 不會生出 context |
| `[3]` | Runtime API 隱式使用 primary context |
| `[4]` | **`cuCtxCreate` 建的 handle 與 primary 位址不同** ← 本題核心 |
| `[5]` | Runtime API 跟隨 context stack 頂端；`cuPointerGetAttribute` 顯示記憶體歸屬 driver ctx |
| `[6]` | context stack 是 push/pop 語意，且 thread-local |
| `[7]` | **有 UVA 時，跨 context 存取其實會成功** |
| `[8]` | 銷毀擁有者 context 後，同一指標變成 illegal memory access |

`[7]` 和 `[8]` 是這支程式最有價值的部分。教材常說「A context 的指標在 B context 完全不能用」——**實測發現這在現代 GPU 上不成立**（本機 UVA=1，跨 context 的 `cudaMemset` 甚至 kernel 存取都成功）。

真正的分界線是**生命週期**：記憶體的命綁在建立它的 context 上，context 一死，指標立刻變野指標。

### 03 需要額外連結 -lcuda

Driver API 在 `libcuda`，由**顯示卡驅動**安裝，不在 CUDA Toolkit 裡。Makefile 已經處理好了。

## 本機環境

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute Capability | 8.9 (Ada Lovelace) |
| SM 數量 | 24 |
| Driver | 580.167.08（支援到 CUDA 13.0） |
| Toolkit | nvcc 12.4 |

換機器跑時數字會不一樣，但**三個結論的方向不會變**（warp 一律是 32）。
