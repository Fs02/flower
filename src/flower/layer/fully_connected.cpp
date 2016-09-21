#include <flower/layer/fully_connected.h>

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
      weights_(Eigen::MatrixXd::Random(definition.input_size(), definition.output_size())),
      bias_(Eigen::MatrixXd::Constant(1, definition.output_size(), 0.0))
{}

void FullyConnected::configure(const IOptimizerDef &optimizer_def)
{
    optimizer_ = optimizer_def.create(net_);
}

Eigen::MatrixXd FullyConnected::forward(const Eigen::MatrixXd &data, bool train)
{
    if (train)
        data_ = data;

    return (data * weights_) + bias_;
}

Eigen::MatrixXd FullyConnected::backward(const Eigen::MatrixXd &errors)
{
    weights_ = optimizer_->optimize(weights_, errors * data_);

    return weights_ * errors;
}
