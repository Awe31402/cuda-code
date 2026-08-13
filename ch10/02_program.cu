// chapter10 片段 2（自 src/chapter10.txt 抽出）
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>
// 定义一个自定义函数对象：对每个元素取平方
struct square
{
    __host__ __device__
    float operator()(const float x) const
    {
        return x * x;
    }
};
// 定义一个自定义函数对象：对两个元素进行加法
struct add_two
{
    __host__ __device__
    float operator()(const thrust::tuple<float, float> &t) const
    {
        return thrust::get<0>(t) + thrust::get<1>(t);
    }
};
int main()
{
    // 创建两个主机向量并初始化
    thrust::host_vector<float> h_vec1(10);
    thrust::host_vector<float> h_vec2(10);
    for (int i = 0; i < 10; i++)
    {
        h_vec1[i] = i + 1.0f;  // 1.0, 2.0, ..., 10.0
        h_vec2[i] = (i + 1.0f) * 2.0f;  // 2.0, 4.0, ..., 20.0
    }
    // 将主机向量复制到设备向量
    thrust::device_vector<float> d_vec1 = h_vec1;
    thrust::device_vector<float> d_vec2 = h_vec2;
    // 使用变换迭代器对第一个向量的元素取平方
    auto square_iter = thrust::make_transform_iterator(d_vec1.begin(), square());
    thrust::device_vector<float> d_squared(10);
    thrust::copy(square_iter, square_iter + 10, d_squared.begin());
    // 打印平方后的结果
    std::cout << "平方操作结果：" << std::endl;
    thrust::copy(d_squared.begin(), d_squared.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    // 使用压缩迭代器对两个向量逐元素相加
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_vec1.begin(), d_vec2.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_vec1.end(), d_vec2.end()));
    thrust::device_vector<float> d_sum(10);
    thrust::transform(zip_begin, zip_end, d_sum.begin(), add_two());
    // 打印相加后的结果
    std::cout << "相加操作结果：" << std::endl;
    thrust::copy(d_sum.begin(), d_sum.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    return 0;
}
