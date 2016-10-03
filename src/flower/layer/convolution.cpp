#include <flower/layer/convolution.h>
#include <iostream>

using namespace flower;

Convolution::Convolution(const Eigen::array<int, 3>& filter_dims, int filter_num, int stride, int padding)
    : ILayerDef(), filter_dims_(filter_dims), filter_num_(filter_num), stride_(stride), padding_(padding)
{}

layer_ptr Convolution::create(Net *net) const
{
    return std::make_shared<ConvolutionLayer>(net, *this);
}


ConvolutionLayer::ConvolutionLayer(Net *net, const Convolution &definition)
    : ILayer(net, definition), filters_(), biases_(),
      stride_(definition.stride()), padding_(definition.padding())
{
    // initialize random filters
    Eigen::Tensor<double, 3> filter(definition.filter_dims()[0], definition.filter_dims()[1], definition.filter_dims()[2]);
    for (int i = 0; i < definition.filter_num(); ++i)
    {
        // TODO: fix duplicated random
        filters_.push_back(filter.random());
        biases_.push_back(0.0);
    }
}

Eigen::Tensor<double, 2> ConvolutionLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    return data;
}

Eigen::Tensor<double, 2> ConvolutionLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    return errors;
}

Eigen::Tensor<double, 4> ConvolutionLayer::forward(const Eigen::Tensor<double, 4> &data, bool train)
{
    Eigen::Tensor<double, 4> result;

    for (unsigned int i = 0; i < filters_.size(); ++i)
    {
        Eigen::Tensor<double, 3> output(data.dimension(0), (data.dimension(2) - filters_[i].dimension(1))/stride_ + 1, (data.dimension(3) - filters_[i].dimension(2))/stride_ + 1);
        convolve(data, output, filters_[i], biases_[i], stride_);
        Eigen::array<long int, 4> result_dims{{output.dimension(0), 1, output.dimension(1), output.dimension(2)}};

        // TODO: concantenate input
        // result = result.concatenate(output.reshape(result_dims), 1);
    }

    return result;
}

Eigen::Tensor<double, 4> ConvolutionLayer::backward(const Eigen::Tensor<double, 4> &errors)
{

}

void ConvolutionLayer::convolve(const Eigen::Tensor<double, 4>& input, Eigen::Tensor<double, 3>& output, const Eigen::Tensor<double, 3>& filter, double bias, int stride)
{
    // TODO: replace with eigen built in convolve
    for (int i = 0; i < input.dimension(0); ++i)
    {
        for (int j = 0; j <= input.dimension(2) - filter.dimension(1); j += stride)
        {
            for (int k = 0; k <= input.dimension(3) - filter.dimension(2); k += stride)
            {
                // perform dot product between local region and kernel
                Eigen::array<long int, 4> offsets = {i, 0, j, k};
                Eigen::array<long int, 4> extents = {1, filter.dimension(0), filter.dimension(1), filter.dimension(2)};
                Eigen::Tensor<double, 3> local = input.slice(offsets, extents).reshape(filter.dimensions());
                Eigen::Tensor<double, 0> dot_product = (local * filter).sum();
                output(i, j/stride, k/stride) = dot_product(0) + bias;
            }
        }
    }
}
