// chapter12 片段 5（自 src/chapter12.txt 抽出）
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
// 常量定义
const double sigma = 1.0;
const double epsilon = 1.0;
const double mass = 1.0;
const double timeStep = 0.001;
const int numParticles = 1000;
const int numSteps = 1000;
// 计算Lennard-Jones势能和作用力
double computeLJPotential(double r) {
    double sr = sigma / r;
    double sr6 = std::pow(sr, 6);
    double sr12 = sr6 * sr6;
    return 4 * epsilon * (sr12 - sr6);
}
double computeLJForce(double r) {
    double sr = sigma / r;
    double sr6 = std::pow(sr, 6);
    double sr12 = sr6 * sr6;
    return 24 * epsilon / r * (2 * sr12 - sr6);
}
// 初始化分子位置和速度
void initializeParticles(std::vector<std::vector<double>> &positions,
                         std::vector<std::vector<double>> &velocities) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 10.0);
    for (int i = 0; i < numParticles; ++i) {
        positions[i][0] = distribution(generator);
        positions[i][1] = distribution(generator);
        positions[i][2] = distribution(generator);
        velocities[i][0] = distribution(generator) - 5.0;
        velocities[i][1] = distribution(generator) - 5.0;
        velocities[i][2] = distribution(generator) - 5.0;
    }
}
// 计算总能量
double computeTotalEnergy(const std::vector<std::vector<double>> &positions,
                          const std::vector<std::vector<double>> &velocities) {
    double totalPotential = 0.0;
    double totalKinetic = 0.0;
    // 计算势能
    for (int i = 0; i < numParticles; ++i) {
        for (int j = i + 1; j < numParticles; ++j) {
            double dx = positions[i][0] - positions[j][0];
            double dy = positions[i][1] - positions[j][1];
            double dz = positions[i][2] - positions[j][2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r > 0.1) {
                totalPotential += computeLJPotential(r);
            }
        }
    }
    // 计算动能
    for (int i = 0; i < numParticles; ++i) {
        double v2 = velocities[i][0] * velocities[i][0] +
                    velocities[i][1] * velocities[i][1] +
                    velocities[i][2] * velocities[i][2];
        totalKinetic += 0.5 * mass * v2;
    }
    return totalPotential + totalKinetic;
}
int main() {
    std::vector<std::vector<double>> positions(numParticles, std::vector<double>(3));
    std::vector<std::vector<double>> velocities(numParticles, std::vector<double>(3));
    initializeParticles(positions, velocities);
    for (int step = 0; step < numSteps; ++step) {
        double totalEnergy = computeTotalEnergy(positions, velocities);
        std::cout << "Step " << step << ", Total Energy: " << totalEnergy << std::endl;
        // 简化的时间积分
        for (int i = 0; i < numParticles; ++i) {
            for (int j = 0; j < 3; ++j) {
                positions[i][j] += velocities[i][j] * timeStep;
            }
        }
    }
    return 0;
}
