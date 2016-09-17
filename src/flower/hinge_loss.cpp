#include <flower/hinge_loss.h>
#include <iostream>

using namespace flower;

HingeLossDef::HingeLossDef(double regularization)
    : ILayerDef(), regularization_(regularization)
{}

layer_ptr HingeLossDef::create(Net *net, const char* name) const
{
    return std::make_shared<HingeLoss>(net, name, *this);
}

HingeLoss::HingeLoss(Net *net, const char* name, const HingeLossDef &definition)
    : ILayer(net, name, definition, 0, 0), regularization_(definition.regularization())
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


Eigen::MatrixXd HingeLoss::forward(const Eigen::MatrixXd &bottom_feat)
{
    return bottom_feat;
}

Eigen::MatrixXd HingeLoss::backward(const Eigen::MatrixXd &top_diff)
{
    return top_diff;
}
