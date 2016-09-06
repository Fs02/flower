#include <flower/fully_connected.h>
#include <flower/feature.h>
#include <iostream>

using namespace flower;

FullyConnectedDef::FullyConnectedDef(const char *name, unsigned int bottom_size, unsigned int top_size)
    : ILayerDef(name), bottom_size_(bottom_size), top_size_(top_size)
{}

FullyConnected::FullyConnected(FullyConnectedDef *definition)
    : ILayer(definition), weights_(0, 0), bias_(0, 0)
{
    // TODO: Merge bias to be single multiplication
    // f(x, W, b) = Wx + b
    weights_ = Feature(Eigen::MatrixXd::Constant(definition->top_size(), definition->bottom_size(), 1.0));
    bias_ = Feature(Eigen::MatrixXd::Constant(definition->top_size(), 1, 0.0));
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
