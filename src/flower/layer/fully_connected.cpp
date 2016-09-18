#include <flower/layer/fully_connected.h>

using namespace flower;

FullyConnectedDef::FullyConnectedDef(unsigned int bottom_size, unsigned int top_size)
    : ILayerDef(), bottom_size_(bottom_size), top_size_(top_size)
{}

layer_ptr FullyConnectedDef::create(Net *net, const char *name) const
{
    return std::make_shared<FullyConnected>(net, name, *this);
}

FullyConnected::FullyConnected(Net* net, const char *name, const FullyConnectedDef &definition)
    : ILayer(net, name, definition), data_(0, 0),
      weights_(Eigen::MatrixXd::Random(definition.bottom_size(), definition.top_size())),
      bias_(Eigen::MatrixXd::Constant(1, definition.top_size(), 0.0))
{}

Eigen::MatrixXd FullyConnected::forward(const Eigen::MatrixXd &data)
{
    data_ = data;
    return (data * weights_) + bias_;
}

Eigen::MatrixXd FullyConnected::backward(const Eigen::MatrixXd &errors)
{
    Eigen::MatrixXd w_update = -0.5 * (errors * data_);
    weights_ += w_update.transpose();

    return weights_ * errors;
}
