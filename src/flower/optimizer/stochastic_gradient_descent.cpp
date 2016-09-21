#include <flower/optimizer/stochastic_gradient_descent.h>

using namespace flower;

optimizer_ptr StochasticGradientDescentDef::instance_;

StochasticGradientDescentDef::StochasticGradientDescentDef(double lr)
    : lr_(lr)
{}

optimizer_ptr StochasticGradientDescentDef::create(Net *net) const
{
    if (!instance_)
        instance_ = std::make_shared<StochasticGradientDescent>(net, *this);

    return instance_;
}

StochasticGradientDescent::StochasticGradientDescent(Net *net, const StochasticGradientDescentDef &definition)
    : IOptimizer(net, definition), lr_(definition.lr())
{}

Eigen::MatrixXd StochasticGradientDescent::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &derivative)
{
    return weight - (lr_ * derivative);
}


