#include <flower/hinge_loss.h>
#include <iostream>

using namespace flower;

flower::HingeLossDef::HingeLossDef(const char *name, double regularization)
    : ILayerDef(name), regularization_(regularization)
{}

HingeLoss::HingeLoss(HingeLossDef *definition)
    : ILayer(definition), regularization_(definition->regularization())
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
