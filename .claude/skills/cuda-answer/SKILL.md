---
name: cuda-answer
description: 用實測回答本 repo 的 CUDA 章節問題。Use when answering a chNN question file, when a CUDA claim needs a number measured on this machine, or when extending an answer's 術語附錄.
---

# 實測回答 CUDA 章節問題

**實測** = 這台機器上跑出來的數字。這個 repo 的答案只接受實測，不接受背誦。

每一個量化斷言背後都要有一條可以重跑的指令。規格書、教科書、你的記憶都不算來源；`./01_device_query` 的輸出才算。

## 步驟

### 1. 列出每一個子問題

問題檔通常一行一題，但一行裡常藏三個子問題（「請說明 X，如何 Y，若 Z 會怎樣」）。先把子問題逐條列出來。

**完成標準：每一個子問題都有對應的章節標題。** 漏掉第三小問是這類題目最常見的失手。

### 2. 為每個量化斷言產生數字

先寫程式或腳本，跑完再寫答案。順序不能反——先寫答案再去找數字佐證，會寫出「看起來對」的東西。

問哪個工具最短：

| 要問的事 | 工具 |
|---|---|
| 硬體規格、上限 | `cudaGetDeviceProperties` |
| kernel 時間 | `cudaEvent`（**不是** `std::chrono`，kernel 啟動是非同步的） |
| context、指標歸屬 | Driver API 的唯讀查詢（`cuCtxGetCurrent`、`cuPointerGetAttribute`） |
| 編譯產物 | `nvcc --dryrun`、`cuobjdump -lelf/-lptx/-sass` |
| 時脈、功耗、P-State | `nvidia-smi --query-gpu=... --format=csv`，連續取樣 |
| 降頻原因 | `nvidia-smi -q -d PERFORMANCE` |

**輸出逐字貼進文件。** 重新排版過的數字讀者無法驗證。

### 3. 重跑任何意外或單次的讀數

意外的讀數是假設，不是事實。**先重跑，再落筆。**

單次取樣尤其危險：`nvidia-smi` 讀到一個 P4 可能只是震盪中的一點，不是穩態。要比較兩種工作型態就背靠背各跑一次，並留意熱歷史這類混淆變因。

**完成標準：這個讀數重現得出來，或者你已經知道它為什麼不重現。**

### 4. 實測與教科書衝突時，實測贏

把修正寫進文件，包含原本的說法錯在哪。讀者需要知道那個流傳很廣的說法不能用。

已經踩過的例子：「A context 的指標在 B context 完全不能用」——有 UVA 時實測會成功，真正的分界是生命週期。

### 5. 量不出來就說量不出來

被雜訊蓋掉、被初始化成本蓋掉、需要沒有的權限——直接寫「量不出差異」並說明原因。

**絕不填一個看起來合理的數字。**

### 6. 需要 root 的操作只報告，不執行

鎖時脈、`-pm`、改 persistence mode 都會動到系統狀態。把指令和預期效果寫進文件，讓使用者自己決定，系統狀態保持原樣。

分析可以建立在已實測到的機制上——例如已經看到 `SW Power Cap: Active`，就能據此推論鎖 P0 的後果。

### 7. 寫答案

見下方「文件慣例」。

### 8. 程式碼歸位

見下方「程式碼慣例」。

### 9. 擴充術語附錄

答案裡出現的 CUDA 術語都要進 `ch01/ch01-questions.md` 的附錄，並附 C/C++ 對照。

附錄最後有一張**常見誤解速查**表。這次實測推翻掉的說法要加一列進去——那是整份文件最有用的一段。

## 文件慣例

答案寫進該章的問答 markdown（ch01 是 [ch01/ch01-questions.md](../../../ch01/ch01-questions.md)）。

- **正體中文**，技術名詞保留英文原文（warp、context、P-State 不翻）
- 讀者是**資深 C/C++ 開發者、CUDA 初學者**：每個 CUDA 概念先給一個 C/C++ 對照（thread≈SIMD lane、context≈行程、Driver API≈dlopen/dlsym）
- 每節**先給結論，再給產生它的程式碼片段**，並說明那幾行為什麼要那樣寫
- 引用程式碼用相對連結指到 `experiments/`
- 每題結尾放一張「想自己驗證？改這幾個地方」表，列出改哪一行會看到什麼；已經跑過的標注「已驗證」

## 程式碼慣例

程式放 `chNN/experiments/`，編號延續該目錄現有的號碼。

- `.cu` 用 `CHECK` / `DRV` 巨集包錯誤檢查
- 開頭註解寫「這支程式在證明什麼」，逐條對應文件章節
- 輸出自帶編號（`[1]`、`[2]`）與解讀，單獨跑也讀得懂
- 進 `Makefile`；執行檔進 `.gitignore`
- 更新 `experiments/README.md`
- 效能實驗要控制變因：固定總工作量、warmup、多次取平均、用擋得住編譯器最佳化的運算（`fmaf` 而非 `+=`）

`make run` 是快的實驗，`make run-tools` 是會加負載的腳本。

## 硬體規格不要寫死在文件慣例裡

本機規格會變（換卡、換驅動）。要數字就跑 `cd ch01/experiments && make && ./01_device_query`，那是唯一來源。

`Makefile` 的 `ARCH` 預設 `sm_89`；換機器用 `make ARCH=sm_XX`。
