# cuda-code

CUDA 學習用 repo，依書本章節分成 `ch01/`~`ch12/`。每章是獨立的範例程式，不是一個整合的專案。

## 這個 repo 的規矩：實測優先

**實測** = 這台機器上跑出來的數字。

回答問題、解釋行為、下效能結論，都要有一條可重跑的指令當依據。規格書與教科書說法只是待驗證的假設——本 repo 已經推翻過其中幾條（例如「跨 context 指標不能用」，有 UVA 時實測會成功）。

量不出來就寫「量不出差異」並說明原因，不要填一個看起來合理的數字。

需要 root 的操作（`nvidia-smi -pm`、`-lgc` 鎖時脈）只報告指令與預期效果，系統狀態保持原樣。

## 章節問答

`chNN/*-questions.md` 是問答文件，`chNN/new_question.md` 是待回答的新題目。

回答這類問題時**用 `cuda-answer` skill**，它有完整的步驟與文件/程式碼慣例。

## 實測程式

`chNN/experiments/` 放為了驗證答案而寫的程式，與章節本身的範例分開。

```bash
cd ch01/experiments
make              # 編譯 .cu，並把 .sh 設為可執行
make run          # 快的實驗（device query、warp、context）
make run-tools    # 會加負載的腳本（nvcc 管線、P-State、nvidia-smi）
make ARCH=sm_XX   # 換機器時指定架構，預設 sm_89
```

每支程式的輸出自帶編號與解讀，單獨跑也讀得懂。執行檔在 `.gitignore` 裡，用 `make` 重建。

**硬體規格不要寫死在文件裡。** 要數字就跑 `./01_device_query`，那是唯一來源。

## 環境

- Toolkit `nvcc` 12.4，Driver 580.167.08（**兩者獨立**；`nvidia-smi` 顯示的 CUDA 版本是驅動最高支援值，不是 toolkit 版本）
- 預設架構 `sm_89`
- 分支 `main`，remote `origin`

## 寫作慣例

問答文件用**正體中文**，技術名詞保留英文（warp、context、P-State 不翻）。讀者設定是**資深 C/C++ 開發者、CUDA 初學者**——每個 CUDA 概念先給一個 C/C++ 對照。

`ch01/ch01-questions.md` 末尾的術語附錄是全 repo 共用的，新術語往那裡加。附錄最後的「常見誤解速查」表收錄被實測推翻的說法。
