// chapter1 片段 6b：驱动API 加载 add.ptx 并启动核函数
// 原文只有 cuModuleLoad / cuModuleGetFunction / cuLaunchKernel 四行片段，
// 这里补全上下文（初始化、显存分配、拷贝、校验）使其可运行。
#include <cuda.h>
#include <iostream>
#include <vector>

#define DRIVER_CHECK(call)                                                     \
    {                                                                          \
        CUresult res = call;                                                   \
        if (res != CUDA_SUCCESS) {                                             \
            const char *name = nullptr;                                        \
            cuGetErrorName(res, &name);                                        \
            std::cerr << "Driver API error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << (name ? name : "unknown") << std::endl;     \
            exit(1);                                                           \
        }                                                                      \
    }

int main(int argc, char **argv) {
    const char *ptxPath = (argc > 1) ? argv[1] : "add.ptx";
    const int size = 1024;
    const size_t bytes = size * sizeof(int);

    std::vector<int> h_a(size), h_b(size), h_c(size, 0);
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    DRIVER_CHECK(cuInit(0));
    DRIVER_CHECK(cuDeviceGet(&cuDevice, 0));
    DRIVER_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    DRIVER_CHECK(cuModuleLoad(&cuModule, ptxPath));
    DRIVER_CHECK(cuModuleGetFunction(&cuFunction, cuModule, "addKernel"));

    CUdeviceptr d_a, d_b, d_c;
    DRIVER_CHECK(cuMemAlloc(&d_a, bytes));
    DRIVER_CHECK(cuMemAlloc(&d_b, bytes));
    DRIVER_CHECK(cuMemAlloc(&d_c, bytes));
    DRIVER_CHECK(cuMemcpyHtoD(d_a, h_a.data(), bytes));
    DRIVER_CHECK(cuMemcpyHtoD(d_b, h_b.data(), bytes));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    int n = size;
    void *args[] = {&d_c, &d_a, &d_b, &n};
    DRIVER_CHECK(cuLaunchKernel(cuFunction, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, 0));
    DRIVER_CHECK(cuCtxSynchronize());
    DRIVER_CHECK(cuMemcpyDtoH(h_c.data(), d_c, bytes));

    std::cout << "驱动API加载PTX结果(前10个): ";
    for (int i = 0; i < 10; ++i) std::cout << h_c[i] << " ";
    std::cout << std::endl;

    bool ok = true;
    for (int i = 0; i < size; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) { ok = false; break; }
    }
    std::cout << (ok ? "校验通过" : "校验失败") << std::endl;

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuCtxDestroy(cuContext);
    return ok ? 0 : 1;
}
