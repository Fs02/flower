#include <flower/softmax_loss.h>
#include <cmath>
#include <iostream>

using namespace flower;

flower::SoftmaxLossDef::SoftmaxLossDef(const char *name)
    : ILayerDef(name)
{}

SoftmaxLoss::SoftmaxLoss(SoftmaxLossDef *definition)
    : ILayer(definition)
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
