#include <flower/layer/sigmoid.h>
#include <iostream>

using namespace flower;

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

SigmoidDef::SigmoidDef(unsigned int size)
    : ILayerDef(), size_(size)
{}


Sigmoid::Sigmoid(Net* net, const char *name, SigmoidDef *definition)
    : ILayer(net, name, definition)
{}

void Sigmoid::forward(Feature &bottom, Feature &top)
{
    auto data = bottom.data().unaryExpr(&sigmoid);
    top.set_data(data);
}

void Sigmoid::backward(Feature &top, Feature &bottom)
{
    // calculate derivative in one pass
    // d = (sigmoid(bottom.data) * (1.0 - sigmoid(bottom.data))) * top.diff
    auto d = bottom.data().unaryExpr([](double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }).cwiseProduct(top.diff());
    bottom.set_diff(bottom.diff() + d);
}
