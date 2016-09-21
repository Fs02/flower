#include <flower/optimizer/rms_prop.h>

using namespace flower;

RmsPropDef::RmsPropDef(double lr, double decay, double eps)
    : lr_(lr), decay_(decay), eps_(eps)
{}

optimizer_ptr RmsPropDef::create(Net *net) const
{
    return std::make_shared<RmsProp>(net, *this);
}

RmsProp::RmsProp(Net *net, const RmsPropDef &definition)
    : IOptimizer(net, definition), lr_(definition.lr()), decay_(definition.decay()), eps_(definition.eps()), gt_(0, 0)
{}

Eigen::MatrixXd RmsProp::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &derivative)
{
    if (gt_.rows() * gt_.cols() == 0)
        gt_= Eigen::ArrayXXd(derivative.rows(), derivative.cols());

    gt_ = decay_ * gt_ + (1.0 - decay_) * derivative.array().pow(2.0);
    return weight - (lr_ * derivative.array() / (gt_.sqrt() + eps_)).matrix();
}
