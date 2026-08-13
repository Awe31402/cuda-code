// chapter1 片段 8：Runtime API 与 Driver API 对比
// 修正说明：原文内嵌的 PTX 字符串无法通过 JIT（寄存器重复定义、
// 缺少 %ntid.x 乘法、s32 与 s64 混用相加、参数名与 ld.param 不匹配）。
// 这里改写为等价且合法的 PTX，其余逻辑保持原样。
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
// 检查CUDA错误宏
#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "      \
                      << cudaGetErrorString(error) << std::endl;               \
            exit(1);                                                           \
        }                                                                      \
    }
#define DRIVER_CHECK(call)                                                     \
    {                                                                          \
        CUresult result = call;                                                \
        if (result != CUDA_SUCCESS)                                            \
        {                                                                      \
            const char *errName = nullptr;                                     \
            cuGetErrorName(result, &errName);                                  \
            std::cerr << "Driver API error at " << __FILE__ << ":"             \
                      << __LINE__ << " -> "                                    \
                      << (errName ? errName : "unknown") << std::endl;         \
            exit(1);                                                           \
        }                                                                      \
    }
// 核函数：运行时API
__global__ void addKernel(int *c, const int *a, const int *b, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}
// 核函数代码字符串：驱动API
const char *ptx_code = R"(
.version 6.4
.target sm_50
.address_size 64

.visible .entry addKernel(
    .param .u64 addKernel_param_0,
    .param .u64 addKernel_param_1,
    .param .u64 addKernel_param_2,
    .param .u32 addKernel_param_3
)
{
    .reg .pred  %p<2>;
    .reg .s32   %r<9>;
    .reg .s64   %rd<11>;

    ld.param.u64 %rd1, [addKernel_param_0];
    ld.param.u64 %rd2, [addKernel_param_1];
    ld.param.u64 %rd3, [addKernel_param_2];
    ld.param.u32 %r1, [addKernel_param_3];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;

    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra $L__end;

    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;
    mul.wide.s32 %rd7, %r5, 4;
    add.s64 %rd8, %rd5, %rd7;
    add.s64 %rd9, %rd6, %rd7;
    ld.global.u32 %r6, [%rd8];
    ld.global.u32 %r7, [%rd9];
    add.s32 %r8, %r6, %r7;
    add.s64 %rd10, %rd4, %rd7;
    st.global.u32 [%rd10], %r8;

$L__end:
    ret;
}
)";
int main()
{
    const int size = 1024;
    const int bytes = size * sizeof(int);
    // 主机内存
    int *h_a = new int[size];
    int *h_b = new int[size];
    int *h_c = new int[size];
    for (int i = 0; i < size; ++i)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    // 运行时API实现
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, size);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    std::cout << "运行时API结果:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // 驱动API实现
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    DRIVER_CHECK(cuInit(0));
    DRIVER_CHECK(cuDeviceGet(&cuDevice, 0));
    DRIVER_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    DRIVER_CHECK(cuModuleLoadDataEx(&cuModule, ptx_code, 0, nullptr, nullptr));
    DRIVER_CHECK(cuModuleGetFunction(&cuFunction, cuModule, "addKernel"));
    CUdeviceptr dd_a, dd_b, dd_c;
    DRIVER_CHECK(cuMemAlloc(&dd_a, bytes));
    DRIVER_CHECK(cuMemAlloc(&dd_b, bytes));
    DRIVER_CHECK(cuMemAlloc(&dd_c, bytes));
    DRIVER_CHECK(cuMemcpyHtoD(dd_a, h_a, bytes));
    DRIVER_CHECK(cuMemcpyHtoD(dd_b, h_b, bytes));
    int n = size;
    void *args[] = {&dd_c, &dd_a, &dd_b, &n};
    DRIVER_CHECK(cuLaunchKernel(cuFunction, blocksPerGrid.x, 1, 1,
                                threadsPerBlock.x, 1, 1, 0, 0, args, 0));
    DRIVER_CHECK(cuCtxSynchronize());
    DRIVER_CHECK(cuMemcpyDtoH(h_c, dd_c, bytes));
    std::cout << "驱动API结果:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;
    cuMemFree(dd_a);
    cuMemFree(dd_b);
    cuMemFree(dd_c);
    cuCtxDestroy(cuContext);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
