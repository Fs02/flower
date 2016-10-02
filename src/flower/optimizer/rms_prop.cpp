#include <flower/optimizer/rms_prop.h>

using namespace flower;

RmsProp::RmsProp(double lr, double decay, double eps)
    : lr_(lr), decay_(decay), eps_(eps)
{}

optimizer_ptr RmsProp::create(Net *net) const
{
    return std::make_shared<RmsPropOptimizer>(net, *this);
}

RmsPropOptimizer::RmsPropOptimizer(Net *net, const RmsProp &definition)
    : IOptimizer(net, definition), lr_(definition.lr()), decay_(definition.decay()), eps_(definition.eps()), gt_(0, 0)
{}

Eigen::Tensor<double, 2> RmsPropOptimizer::optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative)
{
    // initialize gt with derivative dimensions and zero value
    if (gt_.size() == 0)
        gt_ = derivative.constant(0.0);

    gt_ = decay_ * gt_ + (1.0 - decay_) * derivative.pow(2.0);

    return weight - (lr_ * derivative / (gt_.sqrt() + eps_));
}
