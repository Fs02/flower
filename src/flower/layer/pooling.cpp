#include <flower/layer/pooling.h>
#include <iostream>

using namespace flower;

Pooling::Pooling(Mode mode, const Eigen::array<int, 2>& size, int stride)
    : ILayerDef(), mode_(mode), size_(size), stride_(stride)
{}

layer_ptr Pooling::create(Net *net) const
{
    return std::make_shared<PoolingLayer>(net, *this);
}


PoolingLayer::PoolingLayer(Net *net, const Pooling &definition)
    : ILayer(net, definition), mode_(definition.mode()), size_(definition.size()), stride_(definition.stride())
{}

Eigen::Tensor<double, 2> PoolingLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    return data;
}

Eigen::Tensor<double, 2> PoolingLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    return errors;
}

Eigen::Tensor<double, 4> PoolingLayer::forward(const Eigen::Tensor<double, 4> &data, bool train)
{
    Eigen::Tensor<double, 4> output;
    pool(mode_, data, output, size_, stride_);
    return output;
}

Eigen::Tensor<double, 4> PoolingLayer::backward(const Eigen::Tensor<double, 4> &errors)
{

}

void PoolingLayer::pool(Pooling::Mode mode, const Eigen::Tensor<double, 4>& input, Eigen::Tensor<double, 4>& output, const Eigen::array<int, 2>& size, int stride)
{
    for (int i = 0; i < input.dimension(0); ++i)
    {
        for (int j = 0; j < input.dimension(1); ++j)
        {
            for (int k = 0; k <= input.dimension(2) - size[0]; k += stride)
            {
                for (int l = 0; l <= input.dimension(3) - size[1]; l += stride)
                {
                    // perform dot product between local region and kernel
                    Eigen::array<long int, 4> offsets = {i, j, k, l};
                    Eigen::array<long int, 4> extents = {1, 1, size[0], size[1]};
                    Eigen::Tensor<double, 4> local = input.slice(offsets, extents);
                    if (mode == Pooling::Mode::Max) {
                        Eigen::Tensor<double, 0> max_pool = local.maximum();
                        output(i, j, k/stride, l/stride) = max_pool(0);
                    } else {
                        Eigen::Tensor<double, 0> avg_pool = local.mean();
                        output(i, j, k/stride, l/stride) = avg_pool(0);
                    }
                }
            }
        }
    }
}
