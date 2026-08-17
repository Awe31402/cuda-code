#!/usr/bin/env bash
# 04_nvcc_pipeline.sh
#
# 實測 nvcc 的編譯流程、產生的目標碼類型、以及多架構 fatbin。
# ch01-questions.md 第三題的數據來源。
#
# 想證明的事
#   (1) nvcc 是個「分流器」：host 碼給 gcc，device 碼走自己的管線
#   (2) 目標碼有兩種：PTX（虛擬指令集，文字）與 SASS（真實機器碼，在 cubin 裡）
#   (3) -arch=sm_89 其實同時塞了 SASS + PTX
#   (4) 只留 PTX 也能在新卡上跑（驅動 JIT），只留舊 SASS 就會失敗
#
# usage: ./04_nvcc_pipeline.sh

set -u
ARCH=${ARCH:-sm_89}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

hr() { printf '\n=== %s ===\n' "$1"; }

cat > "$WORK/k.cu" <<'EOF'
#include <cstdio>
__global__ void addk(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    float *d; cudaMalloc(&d, 1024);
    addk<<<1, 32>>>(d, d, d, 256);
    cudaDeviceSynchronize();
    printf("kernel result: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaFree(d);
    return 0;
}
EOF

hr "0) 環境"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# ---------------------------------------------------------------- (1) 編譯管線
hr "1) nvcc --dryrun：真正被執行的每一個階段"
echo "（過濾掉環境變數設定行，只留實際指令；只印指令名稱與關鍵參數）"
nvcc -arch=$ARCH --dryrun -o "$WORK/k" "$WORK/k.cu" 2>&1 \
  | grep -v '^#\$ *[A-Z_]*=' | sed 's/^#\$ *//' \
  | awk '{
      tool = $1
      sub(".*/", "", tool)
      if (tool == "rm") next                     # 清暫存檔，不是編譯階段
      key = ""
      for (i = 2; i <= NF; i++) {
        if ($i ~ /^-(E|c|link)$/ || $i ~ /^-arch/) key = key " " $i
      }
      printf "  %-12s%s\n", tool, key
    }'
cat <<'NOTE'

  讀法：
    gcc -E     前處理（host 與 device 各跑一次）
    cudafe++   把 .cu 拆成「host 端 C++」與「device 端」兩半
    cicc       device 編譯器：device 碼 -> PTX
    ptxas      PTX 組譯器：PTX -> cubin（內含 SASS）
    fatbinary  把 cubin + PTX 打包成 fatbin
    gcc -c     編譯 host 端（fatbin 以位元組陣列嵌進去）
    nvlink     device 端連結
    g++        最後連結成執行檔
NOTE

# ---------------------------------------------------------------- (2) 目標碼
hr "2) 目標碼類型：PTX vs SASS"
nvcc -arch=$ARCH -o "$WORK/a_single" "$WORK/k.cu"

echo "-- PTX（虛擬指令集，文字、跨架構）--"
cuobjdump -ptx "$WORK/a_single" | sed -n '/\.visible \.entry/,/^}/p' \
  | grep -E 'ctaid|ntid|tid|mad|setp' | sed 's/^/    /'

echo "-- SASS（$ARCH 真實機器碼，綁死架構）--"
# sed 只吃掉 /* ... */ 註解（位址與機器碼），保留中間的指令本體
cuobjdump -sass "$WORK/a_single" | grep -E 'S2R|IMAD|ISETP' \
  | sed 's|/\*[^*]*\*/||g' | sed 's/^[[:space:]]*/    /;s/[[:space:]]*$//' | head -4

echo
echo "  對照：PTX 的 mad.lo.s32 -> SASS 的 IMAD"
echo "        PTX 用虛擬暫存器 %r1 -> SASS 用真實暫存器 R6"

# ---------------------------------------------------------------- (3) 多架構
hr "3) -arch=$ARCH 到底塞了什麼進去"
echo "-- cubin（SASS）--"; cuobjdump -lelf "$WORK/a_single" | sed 's/^/    /'
echo "-- PTX --";          cuobjdump -lptx "$WORK/a_single" | sed 's/^/    /'
echo
echo "  注意：只寫 -arch=sm_XX，nvcc 會同時放入 SASS 與 PTX。"
echo "        每個架構出現兩個 cubin 是正常的（一個來自編譯、一個來自 device link）。"

hr "4) 多架構 fatbin：用 -gencode 疊出來"
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_89,code=compute_89 \
     -o "$WORK/b_multi" "$WORK/k.cu"
echo "-- cubin（三種架構的 SASS）--"; cuobjdump -lelf "$WORK/b_multi" | sed 's/^/    /'
echo "-- PTX（只留最高的當未來保險）--"; cuobjdump -lptx "$WORK/b_multi" | sed 's/^/    /'

hr "5) 多架構的體積代價（看 .nv_fatbin 區段，避開靜態連結的 cudart）"
for f in a_single b_multi; do
  sz=$(readelf -S "$WORK/$f" 2>/dev/null | grep -A1 'nv_fatbin' | tail -1 | awk '{print strtonum("0x"$1)}')
  printf "  %-10s .nv_fatbin = %6s bytes\n" "$f" "${sz:-?}"
done

# ---------------------------------------------------------------- (4) JIT
hr "6) 只留 PTX，能在本機（$ARCH）上跑嗎？"
nvcc -gencode arch=compute_75,code=compute_75 -o "$WORK/d_ptxonly" "$WORK/k.cu"
echo "  內容：$(cuobjdump -lptx "$WORK/d_ptxonly" | tr -s ' ')"
echo "  沒有任何 cubin（只有 compute_75 的 PTX）"
printf "  執行 -> "; "$WORK/d_ptxonly"
echo "  => 成功。驅動把 PTX 即時編譯（JIT）成 $ARCH 的 SASS。"
echo "     JIT 結果會快取在 ~/.nv/ComputeCache（可用 CUDA_CACHE_DISABLE=1 關閉）"

hr "7) 只留舊架構 SASS、不留 PTX，會怎樣？"
nvcc -gencode arch=compute_75,code=sm_75 -o "$WORK/e_sassonly" "$WORK/k.cu"
echo "  內容：只有 sm_75 的 cubin，沒有 PTX"
printf "  執行 -> "; "$WORK/e_sassonly"
echo "  => 失敗：no kernel image is available for execution on the device"
echo "     這就是最常見的部署錯誤。少了 PTX，就沒有 JIT 的退路。"

hr "結論"
cat <<'EOF'
  1. nvcc 是分流器：host 碼交給系統編譯器，device 碼走 cicc -> ptxas -> fatbinary
  2. 兩種目標碼：PTX（虛擬、可 JIT、向前相容）與 SASS（真實、快、綁死架構）
  3. 支援多架構就用多個 -gencode 疊：
       -gencode arch=compute_75,code=sm_75      # Turing 原生
       -gencode arch=compute_86,code=sm_86      # Ampere 原生
       -gencode arch=compute_89,code=sm_89      # Ada 原生
       -gencode arch=compute_89,code=compute_89 # 留 PTX 給未來的卡
  4. 最後那行 PTX 是關鍵：沒有它，未知的新架構就無法 JIT，直接跑不起來
EOF
