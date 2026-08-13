#!/usr/bin/env bash
# 編譯 ch03~ch12 抽出的所有 CUDA/C++ 程式
# 依 #include 自動附加 -lcurand / -lcublas / -lcufft 等，失敗時再試 -rdc=true
set -u
ARCH=${ARCH:-sm_89}
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/bin"
LOGDIR="$ROOT/build_logs"
mkdir -p "$OUT" "$LOGDIR"

ok=0; failed=0
: > "$LOGDIR/summary.txt"

for src in "$ROOT"/ch{03,04,05,06,07,08,09,10,11,12}/*.cu "$ROOT"/ch12/*.cpp; do
    [ -e "$src" ] || continue
    case "$src" in *_snippet.*) continue ;; esac
    ch=$(basename "$(dirname "$src")")
    name="${ch}_$(basename "${src%.*}")"
    libs=()
    grep -q 'curand'   "$src" && libs+=(-lcurand)
    grep -q 'cublas'   "$src" && libs+=(-lcublas)
    grep -q 'cufft'    "$src" && libs+=(-lcufft)
    grep -q 'cusparse' "$src" && libs+=(-lcusparse)
    grep -q 'cusolver' "$src" && libs+=(-lcusolver)
    grep -q 'nvml.h'   "$src" && libs+=(-lnvidia-ml)
    grep -q 'cuda\.h'  "$src" && libs+=(-lcuda)
    grep -q 'omp\.h'   "$src" && libs+=(-Xcompiler -fopenmp)
    grep -q 'nvToolsExt' "$src" && libs+=(-lnvToolsExt)
    if grep -q 'mpi\.h' "$src"; then
        # 本機 libopenmpi-dev 雖已註冊但 mpi.h 實際不存在，無法編譯
        if [ -z "$(find /usr/include /usr/lib -name mpi.h 2>/dev/null | head -1)" ]; then
            echo "SKIP     $name（系統缺 mpi.h，需 sudo apt install --reinstall libopenmpi-dev）"
            echo "SKIP $name" >> "$LOGDIR/summary.txt"
            continue
        fi
        libs+=($(mpicc --showme:compile 2>/dev/null) $(mpicc --showme:link 2>/dev/null))
    fi

    log="$LOGDIR/$name.log"
    if nvcc -arch="$ARCH" "$src" "${libs[@]}" -o "$OUT/$name" > "$log" 2>&1; then
        echo "OK       $name"; ok=$((ok+1)); echo "OK $name" >> "$LOGDIR/summary.txt"
    elif nvcc -arch="$ARCH" -rdc=true "$src" "${libs[@]}" -lcudadevrt -o "$OUT/$name" >> "$log" 2>&1; then
        echo "OK(rdc)  $name"; ok=$((ok+1)); echo "OK-rdc $name" >> "$LOGDIR/summary.txt"
    else
        echo "FAILED   $name"; failed=$((failed+1)); echo "FAIL $name" >> "$LOGDIR/summary.txt"
    fi
done

echo
echo "成功 $ok / 失敗 $failed（log 在 build_logs/）"
