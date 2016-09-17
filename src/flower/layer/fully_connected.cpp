#include <flower/layer/fully_connected.h>
#include <flower/feature.h>
#include <iostream>

using namespace flower;

FullyConnectedDef::FullyConnectedDef(unsigned int bottom_size, unsigned int top_size)
    : ILayerDef(), bottom_size_(bottom_size), top_size_(top_size)
{}

layer_ptr FullyConnectedDef::create(Net *net, const char *name) const
{
    return std::make_shared<FullyConnected>(net, name, *this);
}

FullyConnected::FullyConnected(Net* net, const char *name, const FullyConnectedDef &definition)
    : ILayer(net, name, definition, definition.top_size(), definition.top_size()), weights_(0, 0), bias_(0, 0),
      weight_feat(0, 0), weight_diff(0, 0), bias_feat(0, 0), bias_diff(0, 0)
{
    // TODO: Merge bias to be single multiplication
    // TODO: Randomize
    // f(x, W, b) = Wx + b
    weights_ = Feature(Eigen::MatrixXd::Constant(definition.top_size(), definition.bottom_size(), 1.0));
    bias_ = Feature(Eigen::MatrixXd::Constant(definition.top_size(), 1, 0.0));

    // TODO
    weight_feat = weights_.data();
    weight_diff = weights_.diff();
    bias_feat = bias_.data();
    bias_diff = bias_.diff();
}

void FullyConnected::forward(Feature &bottom, Feature &top)
{
    // TODO: Bias trick
    top.set_data((weights_.data() * bottom.data()) + bias_.data());
}

void FullyConnected::backward(Feature &top, Feature &bottom)
{
    // TODO: confirm
    weights_.set_diff(top.diff() * bottom.data().transpose());
    bottom.set_diff(weights_.data().transpose() * top.diff());

    // TODO: update weight?
}

const Eigen::MatrixXd &FullyConnected::forward(const Eigen::MatrixXd &bottom_feat)
{
    feat_ = (weight_feat * bottom_feat) + bias_feat;
    return feat_;
}

const Eigen::MatrixXd &FullyConnected::backward(const Eigen::MatrixXd &top_diff)
{
    return top_diff;
}
