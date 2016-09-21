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
    : ILayer(net, name, definition), data_(0, 0),
      weights_(Eigen::MatrixXd::Random(definition.input_size() + 1, definition.output_size())) // with bias
{}

void FullyConnected::configure(const IOptimizerDef &optimizer_def)
{
    optimizer_ = optimizer_def.create(net_);
}

Eigen::MatrixXd FullyConnected::forward(const Eigen::MatrixXd &data, bool train)
{
    if (train)
        data_ = data;

    // append bias
    data_.conservativeResize(data.rows(), data.cols()+1);
    data_.col(data_.cols()-1) = Eigen::VectorXd::Constant(data_.rows(), 1);

    return data_ * weights_;
}

Eigen::MatrixXd FullyConnected::backward(const Eigen::MatrixXd &errors)
{
    weights_ = optimizer_->optimize(weights_, errors * data_);

    Eigen::MatrixXd weight_nobias = weights_;
    weight_nobias.conservativeResize(weight_nobias.rows() - 1, weight_nobias.cols());
    return weight_nobias * errors;
}
