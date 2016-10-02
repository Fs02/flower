#include <flower/optimizer/adam.h>
#include <flower/net.h>
#include <iostream>

using namespace flower;

Adam::Adam(double lr, double beta1, double beta2, double eps)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps)
{}

optimizer_ptr Adam::create(Net *net) const
{
    return std::make_shared<AdamOptimizer>(net, *this);
}

AdamOptimizer::AdamOptimizer(Net *net, const Adam &definition)
    : IOptimizer(net, definition), lr_(definition.lr()), beta1_(definition.beta1()), beta2_(definition.beta2()),
      eps_(definition.eps()), m_(0, 0), v_(0, 0)
{}


Eigen::Tensor<double, 2> AdamOptimizer::optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative)
{
    if (m_.size() == 0)
    {
        m_ = derivative.constant(0.0);
        v_ = derivative.constant(0.0);
    }

    // update first moment
    m_ = beta1_ * m_ + (1.0 - beta1_) * derivative;
    // update second moment
    v_ = beta2_ * v_ + (1.0 - beta2_) * derivative.pow(2.0);

    // bias correction
    m_ = m_ / (1.0 - pow(beta1_, net_->epoch()));
    v_ = v_ / (1.0 - pow(beta2_, net_->epoch()));

    return weight - (lr_ * m_ / (v_.sqrt() + eps_));
}
