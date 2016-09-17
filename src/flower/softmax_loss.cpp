#include <flower/softmax_loss.h>
#include <cmath>
#include <iostream>

using namespace flower;

SoftmaxLossDef::SoftmaxLossDef()
    : ILayerDef()
{}

layer_ptr SoftmaxLossDef::create(Net *net, const char *name) const
{
    return std::make_shared<SoftmaxLoss>(net, name, *this);
}

SoftmaxLoss::SoftmaxLoss(Net *net, const char *name, const SoftmaxLossDef &definition)
    : ILayer(net, name, definition, 0, 0)
{}

void SoftmaxLoss::forward(Feature &bottom, Feature &top)
{
    unsigned int y = 0; // TODO

    // shift the score and do exp operation
    auto exp_score = (bottom.data().array() - bottom.data().maxCoeff()).exp();
    auto loss = -std::log((exp_score / exp_score.sum())(y, 0));
    std::cout << "loss : " << loss;
}

void SoftmaxLoss::backward(Feature &top, Feature &bottom)
{

}

const Eigen::MatrixXd &SoftmaxLoss::forward(const Eigen::MatrixXd &bottom_feat)
{
    return bottom_feat;
}

const Eigen::MatrixXd &SoftmaxLoss::backward(const Eigen::MatrixXd &top_diff)
{
    return top_diff;
}

