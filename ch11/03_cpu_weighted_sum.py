"""chapter11 片段 3（自 src/chapter11.txt 抽出）"""
import numpy as np
import cupy as cp
import time
# 定义向量大小
vector_size = 10**7
cpu_chunk_size = vector_size // 10  # 10%的任务分配给CPU
# 初始化数据
# 修正：原文用 np.random.rand（float64），但下面的 ElementwiseKernel 宣告的是 float32，
# cupy 會丟 "Type is mismatched. a float64 float32"，這裡明確轉成 float32。
vector_a = np.random.rand(vector_size).astype(np.float32)
vector_b = np.random.rand(vector_size).astype(np.float32)
weights = np.random.rand(vector_size).astype(np.float32)
# CPU加权求和函数
def cpu_weighted_sum(a, b, w):
    result = np.zeros_like(a)
    for i in range(len(a)):
        result[i] = a[i] * w[i] + b[i] * (1 - w[i])
    return result
# GPU加权求和核函数
gpu_weighted_sum_kernel = cp.ElementwiseKernel(
    "float32 a, float32 b, float32 w",  # 输入参数
    "float32 result",                   # 输出参数
    "result = a * w + b * (1 - w)",     # 核函数逻辑
    "gpu_weighted_sum_kernel"
)
# 数据划分
cpu_a = vector_a[:cpu_chunk_size]
cpu_b = vector_b[:cpu_chunk_size]
cpu_w = weights[:cpu_chunk_size]
gpu_a = cp.array(vector_a[cpu_chunk_size:])
gpu_b = cp.array(vector_b[cpu_chunk_size:])
gpu_w = cp.array(weights[cpu_chunk_size:])
# CPU计算
start_cpu = time.time()
cpu_result = cpu_weighted_sum(cpu_a, cpu_b, cpu_w)
end_cpu = time.time()
# GPU计算
start_gpu = time.time()
gpu_result = gpu_weighted_sum_kernel(gpu_a, gpu_b, gpu_w)
end_gpu = time.time()
# 合并结果
final_result = np.concatenate([cpu_result, cp.asnumpy(gpu_result)])
# 性能对比输出
print(f"CPU计算时间: {end_cpu - start_cpu:.6f} 秒")
print(f"GPU计算时间: {end_gpu - start_gpu:.6f} 秒")
print(f"总任务结果前10项: {final_result[:10]}")
