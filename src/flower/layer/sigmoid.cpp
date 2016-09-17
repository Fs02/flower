#include <flower/layer/sigmoid.h>
#include <flower/feature.h>
#include <iostream>

using namespace flower;

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

SigmoidDef::SigmoidDef(unsigned int size)
    : ILayerDef(), size_(size)
{}

layer_ptr SigmoidDef::create(Net *net, const char *name) const
{
    return std::make_shared<Sigmoid>(net, name, *this);
//    return new Sigmoid(net, name, this);
}


Sigmoid::Sigmoid(Net* net, const char *name, const SigmoidDef &definition)
    : ILayer(net, name, definition, 1, definition.size())
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

Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd &bottom_feat)
{
    feat_ = bottom_feat;
    return bottom_feat.unaryExpr(&sigmoid);
}

Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd &top_diff)
{
    diff_ = feat_.unaryExpr([](double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }).transpose().cwiseProduct(top_diff);
    return diff_;
}
