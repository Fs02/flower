#include <flower/optimizer/rms_prop.h>
#include <limits>

using namespace flower;

RmsPropDef::RmsPropDef(double learning_rate, double decay_rate)
    : learning_rate_(learning_rate), decay_rate_(decay_rate)
{}

optimizer_ptr RmsPropDef::create(Net *net) const
{
    return std::make_shared<RmsProp>(net, *this);
}

RmsProp::RmsProp(Net *net, const RmsPropDef &definition)
    : IOptimizer(net, definition), learning_rate_(definition.learning_rate()), decay_rate_(definition.decay_rate()), gt_(0, 0)
{}

Eigen::MatrixXd RmsProp::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw)
{
    if (gt_.rows() * gt_.cols() == 0)
        gt_= Eigen::ArrayXXd(dw.rows(), dw.cols());

    gt_ = decay_rate_ * gt_ + (1.0 - decay_rate_) * dw.array().pow(2.0);
    return weight - (learning_rate_ * dw.array() / gt_.sqrt() + std::numeric_limits<double>::epsilon()).matrix().transpose();
}
