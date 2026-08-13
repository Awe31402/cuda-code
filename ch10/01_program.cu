// chapter10 片段 1（自 src/chapter10.txt 抽出）
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <iostream>
// 自定义操作：平方每个元素
struct square
{
    __host__ __device__
    int operator()(const int x) const
    {
        return x * x;
    }
};
int main()
{
    // 第一步：创建主机向量并初始化
    thrust::host_vector<int> h_vec(10);
    thrust::sequence(h_vec.begin(), h_vec.end(), 1); // 填充为1到10的递增序列
    std::cout << "主机向量初始化：" << std::endl;
    for (int i = 0; i < h_vec.size(); i++)
        std::cout << h_vec[i] << " ";
    std::cout << std::endl;
    // 第二步：将数据复制到设备向量
    thrust::device_vector<int> d_vec = h_vec;
    // 第三步：对设备向量排序（降序）
    thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<int>());
    // 复制回主机以验证结果
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    std::cout << "排序后（降序）：" << std::endl;
    for (int i = 0; i < h_vec.size(); i++)
        std::cout << h_vec[i] << " ";
    std::cout << std::endl;
    // 第四步：对设备向量中的元素进行平方操作
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), square());
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    std::cout << "平方操作后：" << std::endl;
    for (int i = 0; i < h_vec.size(); i++)
        std::cout << h_vec[i] << " ";
    std::cout << std::endl;
    // 第五步：归约计算总和
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
    std::cout << "所有元素的总和为：" << sum << std::endl;
    return 0;
}
