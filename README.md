# chapter1~12 程式碼抽取與執行

把 `src/chapter*.txt` 裡的程式碼片段拆成可編譯、可執行的檔案。

## 環境

| 項目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (24 SM, compute capability 8.9) |
| Driver | 580.167.08 |
| nvcc | CUDA 12.4 (V12.4.131) |

## 總覽

| 章節 | 目錄 | 程式數 | 狀態 |
| --- | --- | --- | --- |
| chapter1 | `ch01/` | 6 CUDA + 4 shell + 2 Python | 全數可編譯／執行 |
| chapter2 | `ch02/` | 10 原文 + 2 修正版 | 10 可執行，2 個原文檔在 CUDA 12 編不過（見下） |
| chapter3~12 | `ch03/`~`ch12/` | 74 CUDA/C++ + 4 Python | 71 編譯 OK、3 缺 MPI header 跳過；Python 2 可跑、2 缺 numba |

合計 **87 支可執行檔**，實測 **85 支正常結束**，另 2 支因為本機只有單張 GPU 而主動報錯離開
（`ch04_06_matrix_add`、`ch12_07_compute_forces`，兩者都要求 ≥2 GPU）。

## 編譯與執行

```bash
./build.sh          # chapter1~2，產物在 bin/
./build_all.sh      # chapter3~12，會依 #include 自動加 -lcurand / -lcublas / -lnvToolsExt 等
                    # 兩者都吃 ARCH 環境變數，預設 sm_89：ARCH=sm_75 ./build_all.sh

./run_all.sh                    # 跑 bin/ 下全部，每支限時 90 秒
./run_all.sh ch07 ch08          # 只跑指定章節
TIMEOUT=30 HEAD=20 ./run_all.sh # 調整逾時與顯示行數
```

> `run_all.sh` 用 `head` 截斷輸出，會讓大量列印的程式收到 SIGPIPE 而顯示 `exit=141`。
> 那不是程式錯誤——完整執行（不接管線）這些程式都是 `exit=0`。

驅動 API 範例要在 `bin/` 下執行（會讀同目錄的 `add.ptx`）：

```bash
cd bin && ./ch01_06_driver_load_ptx add.ptx
```

## chapter3~12 的抽取方式

這 10 章共 78 個區塊，用 `tools/extract.py` 依原文的分隔線切塊後自動分類：
判斷是 CUDA / C++ / Python / shell，再從第一個 kernel 或函式名產生檔名
（例如 `ch08/03_warp_reduce_sum_kernel.cu`）。要重跑抽取：

```bash
python3 tools/extract.py
```

各章內容大致是：ch03 記憶體階層、ch04 傳輸與 Unified Memory、ch05 除錯與同步、
ch06 kernel 優化、ch07 合併存取與 bank conflict、ch08 原子操作與歸約、
ch09 CUDA streams、ch10 Thrust/cuRAND/cuBLAS 函式庫、ch11 多 GPU 與 CPU-GPU 協同、
ch12 分子動力學案例。

## 檔案對照

### chapter1

| 檔案 | 對應片段 | 狀態 |
| --- | --- | --- |
| `ch01/01_branch_optimization.cu` | 分支發散／表達式合併片段 | 編譯 OK，執行 OK（原文只是片段，補成完整程式） |
| `ch01/02_env_setup.sh` | CUDA 安裝檢查指令 | 參考用 shell |
| `ch01/03_warp_execution.cu` | Warp 分支模擬 + 暫存器配置測試 | 編譯 OK，執行 OK |
| `ch01/04_nvcc_options.sh` | nvcc 編譯選項 | 參考用 shell |
| `ch01/05_vector_add.cu` | 最簡向量加法 | 編譯 OK，執行 OK |
| `ch01/06_add.cu` + `06_driver_load_ptx.cu` | 「單獨編 PTX + 驅動 API 載入」片段 | 編譯 OK，執行 OK（校驗通過） |
| `ch01/07_env_check.sh` | 驅動／工具鏈版本檢查 | 參考用 shell |
| `ch01/08_runtime_vs_driver.cu` | Runtime API vs Driver API | 編譯 OK，執行 OK（**PTX 已修正**，見下） |
| `ch01/08b_docker_setup.sh` + `Dockerfile` | 驅動調優 / Docker / NVIDIA Container Toolkit | 參考用 shell（含 sudo，未執行） |
| `ch01/09_gpu_monitor.py` | 單卡監控工具 | 執行 OK（設定功耗限制需 root；主迴圈為無限迴圈） |
| `ch01/10_gpu_multi_monitor.py` | 多卡監控工具 | 執行 OK（**已修正 `[N/A]` 解析**） |

### chapter2

| 檔案 | 對應片段 | 狀態 |
| --- | --- | --- |
| `ch02/01_thread_index_workload.cu` | 全域索引 + 工作負載 | OK |
| `ch02/02_thread_lifecycle_group.cu` | 線程生命週期 + Warp 分組 | OK |
| `ch02/03_2d_index.cu` | 二維網格索引映射 | OK |
| `ch02/04_matmul_shared.cu` | 共享記憶體分塊矩陣乘法 | OK，結果與 numpy 比對誤差 0 |
| `ch02/05_block_size_perf.cu` | 不同 block size 效能對比 | OK |
| `ch02/06_dynamic_shared_mem.cu` | 動態共享記憶體 | OK |
| `ch02/07_dynamic_parallel_sum.cu` | 動態並行遞迴求和（原文） | **編譯失敗**，CUDA 12 已移除裝置端 `cudaDeviceSynchronize` |
| `ch02/07_dynamic_parallel_sum_fixed.cu` | 同上修正版 | OK，輸出 136（= 1+2+…+16） |
| `ch02/08_dynamic_parallel_rowsum.cu` | 動態並行遞迴每行平方和（原文） | **編譯失敗**，同上 |
| `ch02/08_dynamic_parallel_rowsum_fixed.cu` | 同上修正版 | 可執行，但演算法本身有問題（見下） |
| `ch02/09_branch_divergence.cu` | 分支發散 vs 分支規約 | OK |
| `ch02/10_warp_shuffle_reduce.cu` | Warp Shuffle 歸約 | OK，輸出 1048576 = 2^20 正確 |

### chapter3~12 值得注意的幾支

| 檔案 | 說明 |
| --- | --- |
| `ch08/08_block_prefix_sum.cu` | 原文執行後印 **"Error in prefix sum computation."**，是真的算錯（見下） |
| `ch08/08_block_prefix_sum_fixed.cu` | 修正版，驗證通過（最後一項 = 1048576） |
| `ch11/05`, `ch11/06`, `ch12/08` | 需要 MPI，本機缺 `mpi.h` 而 **SKIP** |
| `ch11/01`, `ch11/02` | 需要 `numba`，本機未安裝 |
| `ch04/06_matrix_add.cu`, `ch12/07_compute_forces.cu` | 需要 ≥2 GPU，單卡環境下程式自己報錯離開 |
| `ch10/07_matrix_vector_product.cu` | 共軛梯度法跑滿 1000 次迭代才停，可正常收斂輸出 |

## 對原文做的修正

1. **`ch01/08` 內嵌的 PTX 無法 JIT**。原文的 PTX 有四處錯誤：`%rd1`/`%r2` 等暫存器被重複定義、
   全域索引漏乘 `%ntid.x`、`add.s64` 直接加 32-bit 值、`ld.param` 的參數名與 `.param` 宣告不符。
   已改寫成等價且合法的 PTX，程式邏輯不動。
2. **`ch01/06` 核函數加 `extern "C"`**。原文用 `cuModuleGetFunction(..., "addKernel")` 查名字，
   但 C++ 編譯出的 PTX 進入點會被 name mangling 改成 `_Z9addKernel...`，加 `extern "C"` 才找得到。
3. **`ch02/07`、`ch02/08` 的裝置端 `cudaDeviceSynchronize()`**。該 API 在 CUDA 11.6 棄用、
   CUDA 12 移除，所以原文檔案在此環境編不過（錯誤訊息保留在 `bin/*.log`）。
   `_fixed` 版本直接刪掉它——CDP2 語義下父網格結束前會隱式等待子網格，主機端同步一次即可。
4. **`ch01/10` 的 `float("[N/A]")`**。筆記型 GPU 的 `power.limit` 會回報 `[N/A]`，
   原文直接 `float()` 會 `ValueError`，改成容錯解析。
5. **`ch07/05` 缺前置宣告**。原文這段沒有 `#include`、也沒有 `CUDA_CHECK` 定義
   （沿用同章前一段的環境），單獨抽出來會編不過。補上與同章其他段落一致的 include 與巨集。
6. **`ch10/05` 原文被截斷**。文字檔在 `return 0;` 之後就沒了，缺 `main` 的收尾大括號；
   補上 `}` 並加 `<ctime>`（用到 `time()`）。
7. **`ch08/08` 的區塊總和抽取錯誤**。原文寫
   `blockPrefixSum<<<1, gridSize, ...>>>(deviceOutput + blockSize - 1, blockSums, gridSize)`，
   這是從 `deviceOutput[1023]` **連續**讀 1024 個元素，但各區塊的總和實際散落在
   index 1023、2047、3071…（間隔 `blockSize`）。所以 `blockSums` 全錯，
   原文自己的驗證就會印 "Error in prefix sum computation."。
   `_fixed` 版加了一個 `gatherBlockSums` 核函數以 `blockSize` 為 stride 收集，驗證即通過。
8. **`ch11/03`、`ch11/04` 的 cupy 型別不符**。原文用 `np.random.rand`（float64）餵給宣告成
   `float32` 的 `ElementwiseKernel`，執行時丟 `TypeError: Type is mismatched. a float64 float32`，
   加 `.astype(np.float32)` 修正。

## 本機環境缺的東西（非程式問題）

- **MPI header**：`ch11/05`、`ch11/06`、`ch12/08` 需要 `mpi.h`。本機 `libopenmpi-dev` 雖在 dpkg
  裡有登記，但檔案實際不存在（`mpicc` 回報的 include 路徑是空的）。補上即可編譯：
  `sudo apt install --reinstall libopenmpi-dev`，之後 `build_all.sh` 會自動帶入 mpicc 的旗標。
- **numba**：`ch11/01`、`ch11/02` 需要，`pip install numba` 即可（`mpi4py`、`cupy`、`numpy` 都已具備）。
- **第二張 GPU**：`ch04/06`、`ch12/07` 會偵測 GPU 數量並主動報錯離開。

## 原文本身的邏輯問題（保留未改）

- `ch02/07` 註解說「平方和」，實際算的是普通求和。
- `ch02/08` 每層遞迴把 `cols` 減半，但線性索引仍用當前 `cols` 計算，
  等於每層都在讀矩陣的不同區段；最後一層 `cols==1` 的結果會覆蓋前面所有層，
  所以輸出是 `matrix[i]²`（1, 4, 9, 16），而不是整行平方和（204, 816, 1836, 3264）。
  `_fixed` 版把兩者一起印出來對照。
- `ch01/09`、`ch01/10` 的 `monitor_*` 是無限迴圈，且設定功耗限制需要 root，
  在無 sudo 環境下每輪都會印「設定失敗」。
- 所有用 `std::chrono` 毫秒計時的效能測試，在這張卡上核函數都跑在 1 ms 以內，
  一律顯示 `0 ms`，看不出 block size 的差異；要比較的話得換成 `cudaEvent` 或微秒計時。
  ch03~ch06 大量的「未優化 vs 優化」對照多半落在這個問題上。
- `ch02/04` 會把 3 個 256×256 矩陣全部印出來（約 20 萬個數字），建議重定向輸出。
- `ch12/03` 印出的能量全是 0、`ch12/04` 印出的力矩全是 `(0,0,0)`——**這是測資造成的，不是程式壞掉**：
  兩支的位置都初始化成 `(1,1,1)`，前者所有原子重疊使距離為 0、被 `if (dist > 0)` 濾掉；
  後者位置與力平行，外積本來就是零向量。想看到非零結果要換成隨機或格點座標。
- `ch12/02` 的分子受力量級到 1e16，是 Lennard-Jones 的 r⁻¹² 項在隨機座標下幾乎重疊所致，
  原文沒有加最小距離截斷（cutoff）。
- `ch07/03` 兩個版本輸出不同：conflict 版印 `0…15 0…15`，非 conflict 版印 `0…31`。
  原因是 conflict 版的索引寫成 `sharedMemory[tid * 2 % 32]`，32 個執行緒只落在 16 個偶數位置，
  兩兩相撞——既是 2-way bank conflict，也是**寫入競爭**（同一格由兩個執行緒寫、結果未定義）。
  示範 bank conflict 的目的達到了，但順帶引入了 race，讀出來的值不該當成正確結果。
