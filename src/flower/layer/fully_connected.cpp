#include <flower/layer/fully_connected.h>
#include <iostream>

using namespace flower;

FullyConnected::FullyConnected(unsigned int input_size, unsigned int output_size)
    : ILayerDef(), input_size_(input_size), output_size_(output_size)
{}

layer_ptr FullyConnected::create(Net *net) const
{
    return std::make_shared<FullyConnectedLayer>(net, *this);
}

FullyConnectedLayer::FullyConnectedLayer(Net *net, const FullyConnected &definition)
    : ILayer(net, definition), data_(0, 0),
      weights_(definition.input_size() + 1, definition.output_size()) // with bias
{
    weights_.setRandom();
}

void FullyConnectedLayer::configure(const IOptimizerDef &optimizer_def)
{
    optimizer_ = optimizer_def.create(net_);
}

Eigen::Tensor<double, 2> FullyConnectedLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
//    data_ = data;

    // append bias
    Eigen::array<std::pair<int, int>, 2> bias({std::make_pair(0, 0), std::make_pair(0, 1)});
    data_ = data.pad(bias, 1);

    // compute input * weight matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    auto output = data_.contract(weights_, product_dims);

    return output;
}

Eigen::Tensor<double, 2> FullyConnectedLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<Eigen::IndexPair<int>, 1> transpose_product_dims = { Eigen::IndexPair<int>(0, 1) };
    weights_ = optimizer_->optimize(weights_, data_.contract(errors, transpose_product_dims));

    // remove bias from weight
    Eigen::array<int, 2> offsets = {0, 0};
    Eigen::array<int, 2> extents = {(int)weights_.dimension(0) - 1, (int)weights_.dimension(1)};
    Eigen::Tensor<double, 2> weights_nobias = weights_.slice(offsets, extents);

    // compute weight_nobias * errors matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    auto result = weights_nobias.contract(errors, product_dims);

    return result;
}
