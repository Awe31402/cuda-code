#!/usr/bin/env bash
# 逐一執行 bin/ 下的程式，每支限時 TIMEOUT 秒，輸出只留前 N 行
# 用法: ./run_all.sh [ch03 ch04 ...]   不給參數則跑全部
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/bin"
TIMEOUT=${TIMEOUT:-90}
HEAD=${HEAD:-8}

cd "$OUT" || exit 1
filter="${*:-}"
pass=0; fail=0; to=0

for exe in *; do
    [ -x "$exe" ] && [ -f "$exe" ] || continue
    case "$exe" in *.ptx|*.log|*.txt) continue ;; esac
    if [ -n "$filter" ]; then
        match=0
        for f in $filter; do case "$exe" in $f*) match=1 ;; esac; done
        [ "$match" = 1 ] || continue
    fi
    echo "===== $exe ====="
    timeout "$TIMEOUT" "./$exe" 2>&1 | head -"$HEAD"
    rc=${PIPESTATUS[0]}
    if [ "$rc" = 124 ]; then
        echo "[TIMEOUT ${TIMEOUT}s]"; to=$((to+1))
    elif [ "$rc" = 0 ]; then
        echo "[exit=0]"; pass=$((pass+1))
    else
        echo "[exit=$rc]"; fail=$((fail+1))
    fi
    echo
done

echo "===== 統計: 正常 $pass / 非零離開 $fail / 逾時 $to ====="
