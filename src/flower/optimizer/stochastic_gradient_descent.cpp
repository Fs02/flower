#include <flower/optimizer/stochastic_gradient_descent.h>

using namespace flower;

optimizer_ptr StochasticGradientDescentDef::instance_;

StochasticGradientDescentDef::StochasticGradientDescentDef(double learning_rate)
    : learning_rate_(learning_rate)
{}

optimizer_ptr StochasticGradientDescentDef::create(Net *net) const
{
    if (!instance_)
        instance_ = std::make_shared<StochasticGradientDescent>(net, *this);

    return instance_;
}

StochasticGradientDescent::StochasticGradientDescent(Net* net, const StochasticGradientDescentDef &definition)
    : IOptimizer(net, definition), learning_rate_(definition.learning_rate())
{}

Eigen::MatrixXd StochasticGradientDescent::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw)
{
    return weight - (learning_rate_ * dw);
}


