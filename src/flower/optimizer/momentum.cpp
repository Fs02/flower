#include <flower/optimizer/momentum.h>
#include <iostream>

using namespace flower;

Momentum::Momentum(double lr, double mu)
    : lr_(lr), mu_(mu)
{}

optimizer_ptr Momentum::create(Net *net) const
{
    return std::make_shared<MomentumOptimizer>(net, *this);
}

MomentumOptimizer::MomentumOptimizer(Net *net, const Momentum &definition)
    : IOptimizer(net, definition), lr_(definition.lr()), mu_(definition.mu()), vel_(0, 0)
{}

Eigen::Tensor<double, 2> MomentumOptimizer::optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative)
{
    // initialize val with derivative dimensions and zero value
    if (vel_.size() == 0)
        vel_ = derivative.constant(0.0);

    vel_ = (mu_ * vel_) - (lr_ * derivative);

    return weight + vel_;
}
