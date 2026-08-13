"""chapter11 片段 2（自 src/chapter11.txt 抽出）"""
from mpi4py import MPI
import numpy as np
from numba import cuda
# MPI初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# CUDA核函数：矩阵加法
@cuda.jit
def matrix_add_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]
# 矩阵分块和计算函数
def gpu_matrix_addition(local_A, local_B):
    threads_per_block = (16, 16)
    blocks_per_grid_x = (local_A.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (local_A.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    d_A = cuda.to_device(local_A)
    d_B = cuda.to_device(local_B)
    d_C = cuda.device_array_like(local_A)
    matrix_add_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    return d_C.copy_to_host()
# 主逻辑
def main():
    # 矩阵大小
    N = 1024
    M = 1024
    # 每个进程的矩阵分块大小
    local_rows = N // size
    # 进程0初始化全局矩阵并分发
    if rank == 0:
        A = np.random.rand(N, M).astype(np.float32)
        B = np.random.rand(N, M).astype(np.float32)
    else:
        A = None
        B = None
    # 分配本地矩阵块
    local_A = np.empty((local_rows, M), dtype=np.float32)
    local_B = np.empty((local_rows, M), dtype=np.float32)
    comm.Scatter(A, local_A, root=0)
    comm.Scatter(B, local_B, root=0)
    # GPU计算本地块
    local_C = gpu_matrix_addition(local_A, local_B)
    # 收集所有块到进程0
    if rank == 0:
        C = np.empty((N, M), dtype=np.float32)
    else:
        C = None
    comm.Gather(local_C, C, root=0)
    # 进程0验证结果
    if rank == 0:
        C_ref = A + B
        if np.allclose(C, C_ref):
            print("矩阵加法结果正确")
        else:
            print("矩阵加法结果错误")
if __name__ == "__main__":
    main()
