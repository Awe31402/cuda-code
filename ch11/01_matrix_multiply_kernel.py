"""chapter11 片段 1（自 src/chapter11.txt 抽出）"""
import numpy as np
from numba import cuda
# 定义矩阵分块大小
BLOCK_SIZE = 256
# 矩阵乘法核函数
@cuda.jit
def matrix_multiply_kernel(A, B, C, N):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < N and col < N:
        sum = 0
        for k in range(N):
            sum += A[row, k] * B[k, col]
        C[row, col] = sum
# 多GPU矩阵分块乘法
def multi_gpu_matrix_multiply(A, B, C, N, num_gpus):
    # 划分矩阵为子块
    sub_size = N // num_gpus
    streams = []
    devices = []
    
    for i in range(num_gpus):
        cuda.select_device(i)
        streams.append(cuda.stream())
        devices.append(cuda.current_context())
    d_A = [cuda.to_device(A[i * sub_size:(i + 1) * sub_size, :], stream=streams[i]) for i in range(num_gpus)]
    d_B = [cuda.to_device(B, stream=streams[i]) for i in range(num_gpus)]
    d_C = [cuda.device_array((sub_size, N), dtype=np.float32, stream=streams[i]) for i in range(num_gpus)]
    # 启动内核
    for i in range(num_gpus):
        threads_per_block = (16, 16)
        blocks_per_grid = (N // threads_per_block[0], sub_size // threads_per_block[1])
        with streams[i]:
            matrix_multiply_kernel[blocks_per_grid, threads_per_block, streams[i]](d_A[i], d_B[i], d_C[i], N)
    # 将结果复制回主机
    for i in range(num_gpus):
        d_C[i].copy_to_host(C[i * sub_size:(i + 1) * sub_size, :], stream=streams[i])
    # 同步流
    for i in range(num_gpus):
        streams[i].synchronize()
# 测试代码
N = 1024  # 矩阵大小
num_gpus = 2  # 使用的GPU数量
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)
multi_gpu_matrix_multiply(A, B, C, N, num_gpus)
# 验证结果
C_ref = np.dot(A, B)
if np.allclose(C, C_ref):
    print("矩阵乘法结果正确")
else:
    print("矩阵乘法结果错误")
