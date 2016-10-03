#include <flower/layer/convolution.h>

using namespace flower;

inline double d_tanh(double x) { return 1.0 - pow(tanh(x), 2.0); }

ConvolutionDef::ConvolutionDef()
    : ILayerDef()
{}

layer_ptr ConvolutionDef::create(Net *net) const
{
    return std::make_shared<Convolution>(net, *this);
}


Convolution::Convolution(Net *net, const ConvolutionDef &definition)
    : ILayer(net, definition)
{}

Eigen::Tensor<double, 2> Convolution::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    return data;
}

Eigen::Tensor<double, 2> Convolution::backward(const Eigen::Tensor<double, 2> &errors)
{
    return errors;
}

Eigen::Tensor<double, 4> Convolution::forward(const Eigen::Tensor<double, 4> &data, bool train)
{

}

Eigen::Tensor<double, 4> Convolution::backward(const Eigen::Tensor<double, 4> &errors)
{

}

void Convolution::convolve(const Eigen::Tensor<double, 3>& input, const Eigen::Tensor<double, 3>& filter, double bias, Eigen::Tensor<double, 2>& output, int stride)
{
    for (int i = 0; i <= input.dimension(1) - filter.dimension(1); i += stride)
    {
        for (int j = 0; j <= input.dimension(2) - filter.dimension(2); j += stride)
        {
            // perform dot product between local region and kernel
            Eigen::array<long int, 3> offsets = {0, i, j};
            Eigen::array<long int, 3> extents = {3, filter.dimension(1), filter.dimension(2)};
            Eigen::Tensor<double, 0> dot_product = (input.slice(offsets, extents) * filter).sum();
            output(i/stride, j/stride) = dot_product(0) + bias;
        }
    }
}
