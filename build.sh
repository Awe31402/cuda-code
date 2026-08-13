#!/usr/bin/env bash
# 编译 chapter1 / chapter2 抽取出的所有 CUDA 程序
set -u
ARCH=${ARCH:-sm_89}   # RTX 4060 Laptop = compute capability 8.9
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/bin"
mkdir -p "$OUT"

fail=0
build() {  # build <src> <name> [extra nvcc flags...]
    local src="$1" name="$2"; shift 2
    printf '==> %-40s ' "$name"
    if nvcc -arch="$ARCH" "$@" "$src" -o "$OUT/$name" 2> "$OUT/$name.log"; then
        echo "OK"
    else
        echo "FAILED (见 bin/$name.log)"
        fail=$((fail + 1))
    fi
}

build "$ROOT/ch01/01_branch_optimization.cu"  ch01_01_branch_optimization
build "$ROOT/ch01/03_warp_execution.cu"       ch01_03_warp_execution
build "$ROOT/ch01/05_vector_add.cu"           ch01_05_vector_add
build "$ROOT/ch01/06_driver_load_ptx.cu"      ch01_06_driver_load_ptx -lcuda
build "$ROOT/ch01/08_runtime_vs_driver.cu"    ch01_08_runtime_vs_driver -lcuda
# 06 的核函数需先生成 PTX 供驱动API加载
printf '==> %-40s ' "ch01_06 add.ptx"
if nvcc -arch="$ARCH" -ptx "$ROOT/ch01/06_add.cu" -o "$OUT/add.ptx" 2> "$OUT/add.ptx.log"; then
    echo "OK"
else
    echo "FAILED"; fail=$((fail + 1))
fi

build "$ROOT/ch02/01_thread_index_workload.cu"      ch02_01_thread_index_workload
build "$ROOT/ch02/02_thread_lifecycle_group.cu"     ch02_02_thread_lifecycle_group
build "$ROOT/ch02/03_2d_index.cu"                   ch02_03_2d_index
build "$ROOT/ch02/04_matmul_shared.cu"              ch02_04_matmul_shared
build "$ROOT/ch02/05_block_size_perf.cu"            ch02_05_block_size_perf
build "$ROOT/ch02/06_dynamic_shared_mem.cu"         ch02_06_dynamic_shared_mem
# 动态并行需要可重定位设备代码 + 链接 cudadevrt
build "$ROOT/ch02/07_dynamic_parallel_sum.cu"       ch02_07_dynamic_parallel_sum -rdc=true -lcudadevrt
build "$ROOT/ch02/08_dynamic_parallel_rowsum.cu"    ch02_08_dynamic_parallel_rowsum -rdc=true -lcudadevrt
build "$ROOT/ch02/07_dynamic_parallel_sum_fixed.cu"    ch02_07_dynamic_parallel_sum_fixed -rdc=true -lcudadevrt
build "$ROOT/ch02/08_dynamic_parallel_rowsum_fixed.cu" ch02_08_dynamic_parallel_rowsum_fixed -rdc=true -lcudadevrt
build "$ROOT/ch02/09_branch_divergence.cu"          ch02_09_branch_divergence
build "$ROOT/ch02/10_warp_shuffle_reduce.cu"        ch02_10_warp_shuffle_reduce

echo
echo "编译失败数量: $fail"
exit 0
