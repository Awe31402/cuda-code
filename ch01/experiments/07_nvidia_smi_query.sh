#!/usr/bin/env bash
# 07_nvidia_smi_query.sh
#
# nvidia-smi 查詢 GPU 即時狀態的各種寫法。
# ch01-questions.md 第五題的數據來源，重點在「輸出 GPU0 的即時功耗與記憶體使用」。
#
# usage: ./07_nvidia_smi_query.sh
#        SHOW_LOAD=1 ./07_nvidia_smi_query.sh   # 順便開負載，數字比較好看

set -u
hr() { printf '\n=== %s ===\n' "$1"; }

# 想看有負載的數字就開一個背景負載
LOADPID=""
if [ "${SHOW_LOAD:-0}" = "1" ] && [ -x ./05_gpu_load ]; then
    ./05_gpu_load 40 2048 >/dev/null 2>&1 &
    LOADPID=$!
    sleep 3
    echo "（背景負載已啟動，pid=$LOADPID）"
fi
cleanup() { [ -n "$LOADPID" ] && kill "$LOADPID" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

hr "1) 最基本：整張表"
echo "  \$ nvidia-smi"
nvidia-smi | head -12 | sed 's/^/  /'
echo "  （人看很方便，但欄位固定、不好用程式解析）"

hr "2) 本題要的答案：GPU0 的即時功耗與記憶體使用"
echo "  \$ nvidia-smi -i 0 --query-gpu=power.draw,memory.used,memory.total,memory.free --format=csv"
nvidia-smi -i 0 --query-gpu=power.draw,memory.used,memory.total,memory.free --format=csv | sed 's/^/  /'
echo
echo "  -i 0        只看 GPU 0"
echo "  --query-gpu 指定要哪些欄位（逗號分隔，不能有空格）"
echo "  --format    csv / noheader（不要標題）/ nounits（不要單位）"

hr "3) 拿掉標題與單位，方便腳本處理"
echo "  \$ nvidia-smi -i 0 --query-gpu=power.draw,memory.used --format=csv,noheader,nounits"
nvidia-smi -i 0 --query-gpu=power.draw,memory.used --format=csv,noheader,nounits | sed 's/^/  /'
echo
echo "  例：只取功耗數字餵給別的程式"
w=$(nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits)
echo "    目前功耗 = $w W"

hr "4) 每秒持續取樣（-l 1），這裡取 4 筆"
echo "  \$ nvidia-smi -i 0 --query-gpu=timestamp,power.draw,memory.used,utilization.gpu --format=csv -l 1"
timeout 5 nvidia-smi -i 0 \
  --query-gpu=timestamp,power.draw,memory.used,utilization.gpu \
  --format=csv -l 1 | sed 's/^/  /'

hr "5) 一次看完整的即時狀態（實務上最常用的一行）"
echo "  \$ nvidia-smi -i 0 --query-gpu=pstate,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,power.default_limit,memory.used,memory.total,temperature.gpu --format=csv"
nvidia-smi -i 0 --query-gpu=pstate,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,power.default_limit,memory.used,memory.total,temperature.gpu \
  --format=csv | sed 's/^/  /'

hr "6) dmon：緊湊的滾動監控（適合長時間看趨勢）"
echo "  \$ nvidia-smi dmon -s puct -c 4"
echo "    s=選擇欄位群組：p 功耗溫度 / u 使用率 / c 時脈 / t PCIe 流量"
timeout 8 nvidia-smi dmon -s puct -c 4 2>/dev/null | sed 's/^/  /'

hr "7) 哪個 process 吃掉顯存"
echo "  \$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
out=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv)
if [ "$(echo "$out" | wc -l)" -le 1 ]; then
    echo "  $out"
    echo "  （目前沒有 compute process；用 SHOW_LOAD=1 重跑就看得到）"
else
    echo "$out" | sed 's/^/  /'
fi

hr "8) 完整欄位清單去哪查"
echo "  \$ nvidia-smi --help-query-gpu        # 每個欄位的說明"
echo "  \$ nvidia-smi -q                     # 傾印全部資訊（很長）"
echo "  \$ nvidia-smi -q -d POWER,MEMORY,CLOCK,TEMPERATURE   # 只看某幾類"
echo
echo "  本機可查欄位數：$(nvidia-smi --help-query-gpu | grep -c '^ *"') 個左右"

hr "常用欄位速查"
cat <<'EOF'
  功耗    power.draw            目前功耗（W）
          power.default_limit   預設功耗上限
          power.max_limit       硬體最大上限
  記憶體  memory.used/free/total  顯存使用/剩餘/總量（MiB）
          utilization.memory    記憶體「頻寬」使用率，不是佔用率！
  運算    utilization.gpu       有 kernel 在跑的時間佔比（不是算力使用率）
  時脈    clocks.sm / clocks.mem / clocks.max.sm
  狀態    pstate                P0（最高效能）~ P12（最省電）
  溫度    temperature.gpu

  兩個容易誤解的欄位：
    utilization.gpu     = 「這段時間有多少比例在跑 kernel」。
                          就算只用了 1 個 thread 也可能顯示 100%。
    utilization.memory  = 記憶體頻寬忙碌程度，跟 memory.used 完全不同。
EOF
