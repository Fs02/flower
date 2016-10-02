#include <flower/layer/fully_connected.h>
#include <iostream>

using namespace flower;

FullyConnectedDef::FullyConnectedDef(unsigned int input_size, unsigned int output_size)
    : ILayerDef(), input_size_(input_size), output_size_(output_size)
{}

layer_ptr FullyConnectedDef::create(Net *net, const char *name) const
{
    return std::make_shared<FullyConnected>(net, name, *this);
}

FullyConnected::FullyConnected(Net *net, const char *name, const FullyConnectedDef &definition)
    : ILayer(net, name, definition), data_(0, 0), input_(0, 0),
      parameters_(definition.input_size() + 1, definition.output_size()), // with bias
      weights_(Eigen::MatrixXd::Random(definition.input_size() + 1, definition.output_size())) // with bias
{
    parameters_.setRandom();
}

void FullyConnected::configure(const IOptimizerDef &optimizer_def)
{
    optimizer_ = optimizer_def.create(net_);
}

Eigen::MatrixXd FullyConnected::forward(const Eigen::MatrixXd &data, bool train)
{
    data_ = data;

    // append bias
    data_.conservativeResize(data.rows(), data.cols()+1);
    data_.col(data_.cols()-1) = Eigen::VectorXd::Constant(data_.rows(), 1);

    return data_ * weights_;
}

Eigen::MatrixXd FullyConnected::backward(const Eigen::MatrixXd &errors)
{
    weights_ = optimizer_->optimize(weights_, (errors * data_).transpose());

    Eigen::MatrixXd weight_nobias = weights_;
    weight_nobias.conservativeResize(weight_nobias.rows() - 1, weight_nobias.cols());
    return weight_nobias * errors;
}


Eigen::Tensor<double, 2> FullyConnected::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    input_ = data;

    // append bias
    Eigen::array<std::pair<int, int>, 2> bias({std::make_pair(0, 0), std::make_pair(0, 1)});
    input_ = data.pad(bias, 1);

    // compute input * weight matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    auto output = input_.contract(parameters_, product_dims);

    return output;
}

Eigen::Tensor<double, 2> FullyConnected::backward(const Eigen::Tensor<double, 2> &errors)
{
    // TODO: update weight

    // remove bias from weight
    Eigen::array<int, 2> offsets = {0, 0};
    Eigen::array<int, 2> extents = {(int)parameters_.dimension(0) - 1, (int)parameters_.dimension(1)};
    Eigen::Tensor<double, 2> parameters_nobias = parameters_.slice(offsets, extents);

    // compute weight_nobias * errors matrix product
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    auto result = parameters_nobias.contract(errors, product_dims);

    return result;
}
