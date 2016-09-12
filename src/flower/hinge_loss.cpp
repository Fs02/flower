#include <flower/hinge_loss.h>
#include <iostream>

using namespace flower;

HingeLossDef::HingeLossDef(double regularization)
    : ILayerDef(), regularization_(regularization)
{}

ILayer *HingeLossDef::create(Net *net, const char* name)
{
    return new HingeLoss(net, name, this);
}

HingeLoss::HingeLoss(Net *net, const char* name, HingeLossDef *definition)
    : ILayer(net, name, definition), regularization_(definition->regularization())
{}

void HingeLoss::forward(Feature &bottom, Feature &top)
{
    unsigned int y = 0; // TODO
    double y_value = bottom.data()(y, 0);

    auto loss = (bottom.data().array() - y_value + 1.0).cwiseMax(0).sum() - 1.0;
    std::cout << loss;
}

void HingeLoss::backward(Feature &top, Feature &bottom)
{

}
