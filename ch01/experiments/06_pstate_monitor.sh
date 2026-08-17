#!/usr/bin/env bash
# 06_pstate_monitor.sh
#
# 實測 GPU 的 P-State 怎麼隨負載動態調整。
# ch01-questions.md 第四題的數據來源。
#
# 想證明的事
#   (1) 閒置時停在低效能狀態（本機 P5），有負載幾乎立刻跳到 P0
#   (2) P0 不等於「時脈最高」—— 本機 P0 的 SM 時脈反而比 P5 低
#   (3) 真正被拉上去的是「記憶體時脈」（810 -> 8001 MHz）
#   (4) P0 之下實際時脈由功耗上限決定（SW Power Cap 會顯示 Active）
#
# 需要 05_gpu_load（先 make）
# usage: ./06_pstate_monitor.sh

set -u
LOAD=./05_gpu_load
[ -x "$LOAD" ] || { echo "找不到 $LOAD，請先執行 make"; exit 1; }

FIELDS=pstate,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu

hr() { printf '\n=== %s ===\n' "$1"; }

head_row() {
    printf "%-8s %-7s %-6s %-9s %-9s %-8s %-5s\n" \
        "$1" "pstate" "util%" "sm_MHz" "mem_MHz" "power_W" "temp"
}

# 取樣一次，第一個參數是要印在最左邊的標籤
sample() {
    nvidia-smi --query-gpu=$FIELDS --format=csv,noheader,nounits \
      | awk -F', ' -v tag="$1" \
        '{printf "%-8s %-7s %-6s %-9s %-9s %-8s %-5s\n", tag, $1, $2, $3, $4, $5, $6}'
}

hr "0) 硬體上限（先知道天花板在哪，才看得懂後面的數字）"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.default_limit,power.max_limit \
           --format=csv | sed 's/^/  /'

hr "1) 閒置狀態"
head_row "t"
for i in 1 2 3; do sample "${i}s"; sleep 1; done

hr "2) 加上 compute-bound 負載（16 MB 工作集），每 0.5 秒取樣"
$LOAD 12 16 >/dev/null 2>&1 &
head_row "t"
for i in $(seq 0 23); do
    sample "$(echo "scale=1; $i*0.5" | bc)"
    sleep 0.5
done
wait

hr "3) 負載中的時脈受限原因（關鍵證據）"
$LOAD 8 16 >/dev/null 2>&1 &
sleep 3
nvidia-smi -q -d PERFORMANCE \
  | sed -n '/Performance State/,/Clocks Event Reasons Counters/p' \
  | grep -E 'Performance State|Power Cap|Thermal|HW Slowdown|Idle' | sed 's/^/  /'
wait

hr "4) 記憶體吃重的負載（2 GB 工作集）拿來對照
   注意：此時 GPU 已被前面的測試烘熱，數字含熱歷史影響，
   要做嚴謹的工作型態比較請冷機後單獨跑"
$LOAD 8 2048 >/dev/null 2>&1 &
sleep 3
head_row "t"
for i in 1 2 3; do sample "${i}s"; sleep 1; done
wait

hr "5) 負載結束後多久回落"
head_row "t"
for i in 1 2 3 4; do sleep 2; sample "+$((i*2))s"; done

hr "怎麼解讀"
cat <<'EOF'
  P-State 是「效能狀態」，不是「時脈」。P0 最高、數字越大越省電。
  它一次調整整組電壓/頻率設定（SM、記憶體、視訊等各個 clock domain）。

  本機觀察到的重點：
    閒置 P5 : SM 2475 MHz、記憶體時脈很低（810 MHz）、約 17 W、約 60 度
    負載 P0 : 記憶體時脈拉到滿（8001 MHz），SM 反而降到 ~1950 MHz、約 45 W
    上限    : SM 最高 3105 MHz —— 閒置與負載「都沒有」跑到上限

  (1) 反應很快：負載一進來，0.5 秒內就從 P5 跳到 P0；
      負載結束後約 4 秒回到 P5。

  (2) P0 的 SM 時脈比 P5 低，看起來很反直覺。原因是：
      - 閒置時沒有實際工作，高時脈幾乎不花電，讀數接近 boost 值
      - 一旦真的有工作，功耗限制立刻生效（上面 [3] 顯示 SW Power Cap: Active）
      - 所以 P0 的意思是「允許用到最高效能狀態」，
        不是「保證跑在最高時脈」。實際時脈由 boost 演算法在
        功耗與溫度預算內動態決定。

  (3) 真正被 P-State 拉上去的是「記憶體時脈」：810 -> 8001 MHz，約 10 倍。
      P-State 是一整組 clock domain 的設定，不是單一數字。

  (4) 持續負載下狀態會震盪：可以看到 P0/P3/P4 交替出現，
      記憶體時脈跟著 8001/7001/6001 降階。GPU 一直在重新協商
      功耗與溫度預算，並不是「跳到 P0 就固定不動」。

  想強制鎖在 P0 需要 root：
    sudo nvidia-smi -pm 1                      # persistence mode
    sudo nvidia-smi -lgc <min>,<max>           # 鎖 SM 時脈
    sudo nvidia-smi --lock-memory-clocks=<v>   # 鎖記憶體時脈
    sudo nvidia-smi -rgc                       # 解除
  本腳本刻意不去改系統狀態。
EOF
