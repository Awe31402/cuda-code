"""chapter11 片段 4（自 src/chapter11.txt 抽出）"""
import numpy as np
import cupy as cp
import threading
import time
# 初始化数据
matrix_size = (10000, 10000)
cpu_chunk_size = 2000  # 分配给CPU的矩阵块大小
# 修正：原文用 np.random.rand（float64），但 ElementwiseKernel 宣告的是 float32，
# cupy 會丟 "Type is mismatched. a float64 float32"，這裡明確轉成 float32。
matrix_a = np.random.rand(*matrix_size).astype(np.float32)
matrix_b = np.random.rand(*matrix_size).astype(np.float32)
weights = np.random.rand(*matrix_size).astype(np.float32)
# CPU计算函数
def cpu_compute(a, b, w, result):
    for i in range(a.shape[0]):
        result[i, :] = a[i, :] * w[i, :] + b[i, :] * (1 - w[i, :])
# GPU计算核函数
gpu_kernel = cp.ElementwiseKernel(
    "float32 a, float32 b, float32 w",
    "float32 result",
    "result = a * w + b * (1 - w)",
    "gpu_compute"
)
# 数据划分
cpu_a = matrix_a[:cpu_chunk_size, :]
cpu_b = matrix_b[:cpu_chunk_size, :]
cpu_w = weights[:cpu_chunk_size, :]
gpu_a = cp.array(matrix_a[cpu_chunk_size:, :])
gpu_b = cp.array(matrix_b[cpu_chunk_size:, :])
gpu_w = cp.array(weights[cpu_chunk_size:, :])
# 结果存储
cpu_result = np.zeros((cpu_chunk_size, matrix_size[1]))
gpu_result = cp.zeros((matrix_size[0] - cpu_chunk_size, matrix_size[1]))
# 开始计时
start_time = time.time()
# CPU线程
cpu_thread = threading.Thread(target=cpu_compute, args=(cpu_a, cpu_b, cpu_w, cpu_result))
cpu_thread.start()
# GPU计算
gpu_result = gpu_kernel(gpu_a, gpu_b, gpu_w)
# 等待CPU线程完成
cpu_thread.join()
# 合并结果
final_result = np.vstack([cpu_result, cp.asnumpy(gpu_result)])
# 结束计时
end_time = time.time()
# 输出结果和性能信息
print(f"总计算时间: {end_time - start_time:.6f} 秒")
print(f"结果矩阵前5行:\n{final_result[:5, :5]}")
