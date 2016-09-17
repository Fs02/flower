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

int FullyConnected::count = 0;

FullyConnected::FullyConnected(Net* net, const char *name, const FullyConnectedDef &definition)
    : ILayer(net, name, definition, definition.top_size(), definition.top_size()), weights_(0, 0), bias_(0, 0),
      weight_feat(0, 0), weight_diff(0, 0), bias_feat(0, 0), bias_diff(0, 0)
{
    // TODO: Merge bias to be single multiplication
    // TODO: Randomize
    // f(x, W, b) = Wx + b
    weights_ = Feature(Eigen::MatrixXd::Constant(definition.bottom_size(), definition.top_size(), 1.0));
    bias_ = Feature(Eigen::MatrixXd::Constant(1, definition.top_size(), 0.0));

    // TODO
//    weight_feat = weights_.data();
    weight_diff = weights_.diff();
//    bias_feat = bias_.data();
    bias_diff = bias_.diff();

    if (count == 0) {
        weight_feat = Eigen::MatrixXd(2, 2);
        bias_feat = Eigen::MatrixXd(1, 2);

        weight_feat << 0.15, 0.25, 0.2, 0.3;
        bias_feat << 0.35, 0.35;
    } else if (count == 1) {
        weight_feat = Eigen::MatrixXd(2, 2);
        bias_feat = Eigen::MatrixXd(1, 2);

        weight_feat << 0.40, 0.50, 0.45, 0.55;
        bias_feat << 0.60, 0.60;
    } else {
        weight_diff = weights_.diff();
        bias_diff = bias_.diff();
    }

    std::cout << "\ncount" << ++count;
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

Eigen::MatrixXd FullyConnected::forward(const Eigen::MatrixXd &bottom_feat)
{
    std::cout << "weights_\n" << weight_feat;
    std::cout << "\nbottom_";
    feat_ = bottom_feat;
    return (bottom_feat * weight_feat) + bias_feat;
}

Eigen::MatrixXd FullyConnected::backward(const Eigen::MatrixXd &top_diff)
{
    diff_ = weight_feat * top_diff;
    Eigen::MatrixXd w_grad = -0.5 * (top_diff * feat_);

    weight_feat = weight_feat + w_grad.transpose();

    std::cout << "\nWEIGHT\n"
              << weight_feat
              << "\nAAA\n";

    return diff_;
}
