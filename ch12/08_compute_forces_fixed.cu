// chapter12 片段 8（修正版）：MPI + CUDA 多進程分子受力計算
//
// 原文有三個問題，導致結果會隨 rank 數改變（np=2 的力約為 np=1 的一半）：
//
//   1. 每個 rank 只配置並計算「自己那一份」粒子，rank 之間完全不交換座標。
//      N-body 受力必須讓每顆粒子看到所有其他粒子（或至少 cutoff 內的所有粒子），
//      原文既沒有 all-gather 也沒有 halo 交換，所以每顆粒子只感受到 1/worldSize 的鄰居。
//   2. 每個 rank 都用未 srand 的 rand()，序列完全相同 → 所有 rank 拿到「同一批」座標，
//      MPI_Gather 收回來的是重複資料，根本不是有效的區域分解。
//   3. cudaSetDevice(worldRank) 沒檢查回傳值。單 GPU 機器上 rank 1 會拿到
//      cudaErrorInvalidDevice，然後靜默沿用 device 0——剛好能跑，但錯誤被吞掉了。
//
// 修正：rank 0 產生全部座標後 MPI_Bcast 給所有人；每個 rank 對「全域座標」計算
// 自己負責那段粒子的受力；cudaSetDevice 依實際 GPU 數取模並檢查錯誤。
// 這樣結果與 worldSize 無關，程式會自我驗證這一點。
#include <cuda_runtime.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    }

// GPU内核函数：計算 [myStart, myStart+myCount) 這段粒子對「全部」粒子的受力
__global__ void computeForces(const float *positions, float *forces,
                              int numParticles, int myStart, int myCount, float cutoff) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local < myCount) {
        int idx = myStart + local;              // 全域粒子編號
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        float xi = positions[idx * 3];
        float yi = positions[idx * 3 + 1];
        float zi = positions[idx * 3 + 2];
        for (int j = 0; j < numParticles; j++) {   // 對全體粒子求和
            if (j == idx) continue;
            float xj = positions[j * 3];
            float yj = positions[j * 3 + 1];
            float zj = positions[j * 3 + 2];
            float dx = xj - xi;
            float dy = yj - yi;
            float dz = zj - zi;
            float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < cutoff * cutoff) {
                float dist = sqrtf(distSq);
                float force = (1.0f / (dist * dist + 1e-6f));
                fx += force * dx / dist;
                fy += force * dy / dist;
                fz += force * dz / dist;
            }
        }
        forces[local * 3] = fx;
        forces[local * 3 + 1] = fy;
        forces[local * 3 + 2] = fz;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    const int numParticles = 10000;
    const int blockSize = 256;
    const float cutoff = 1.0f;

    if (numParticles % worldSize != 0) {
        if (worldRank == 0)
            std::cerr << "為求簡單，numParticles 需能被 rank 數整除" << std::endl;
        MPI_Finalize();
        return 1;
    }
    const int myCount = numParticles / worldSize;
    const int myStart = worldRank * myCount;

    // 1. rank 0 產生全部座標，再廣播給所有 rank（原文缺這步）
    std::vector<float> allPositions(numParticles * 3);
    if (worldRank == 0) {
        srand(12345);   // 固定種子，方便不同 rank 數之間比對
        for (int i = 0; i < numParticles * 3; i++) {
            allPositions[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    MPI_Bcast(allPositions.data(), numParticles * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 2. 依實際 GPU 數取模，並檢查錯誤（原文直接用 worldRank 且不檢查）
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (worldRank == 0) std::cerr << "找不到 CUDA 裝置" << std::endl;
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(worldRank % deviceCount));

    float *d_positions, *d_forces;
    CUDA_CHECK(cudaMalloc(&d_positions, numParticles * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_forces, myCount * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_positions, allPositions.data(),
                          numParticles * 3 * sizeof(float), cudaMemcpyHostToDevice));

    int gridSize = (myCount + blockSize - 1) / blockSize;
    computeForces<<<gridSize, blockSize>>>(d_positions, d_forces,
                                           numParticles, myStart, myCount, cutoff);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> myForces(myCount * 3);
    CUDA_CHECK(cudaMemcpy(myForces.data(), d_forces,
                          myCount * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> globalForces;
    if (worldRank == 0) globalForces.resize(numParticles * 3);
    MPI_Gather(myForces.data(), myCount * 3, MPI_FLOAT,
               worldRank == 0 ? globalForces.data() : nullptr, myCount * 3,
               MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        std::cout << "worldSize = " << worldSize
                  << "（結果應與 rank 數無關）" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "Particle " << i << ": Force = ("
                      << globalForces[i * 3] << ", "
                      << globalForces[i * 3 + 1] << ", "
                      << globalForces[i * 3 + 2] << ")" << std::endl;
        }
        double checksum = 0.0;
        for (int i = 0; i < numParticles * 3; i++) checksum += globalForces[i];
        std::cout << "全部受力分量總和 checksum = " << checksum << std::endl;
    }

    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_forces));
    MPI_Finalize();
    return 0;
}
