#include <flower/optimizer/momentum.h>

using namespace flower;

MomentumDef::MomentumDef(double lr, double mu)
    : lr_(lr), mu_(mu)
{}

optimizer_ptr MomentumDef::create(Net *net) const
{
    return std::make_shared<Momentum>(net, *this);
}

Momentum::Momentum(Net *net, const MomentumDef &definition)
    : IOptimizer(net, definition), lr_(definition.lr()), mu_(definition.mu()), velocity_(0, 0)
{}

Eigen::MatrixXd Momentum::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw)
{
    if (velocity_.rows() * velocity_.cols() == 0)
        velocity_ = Eigen::MatrixXd(dw.rows(), dw.cols());

    velocity_ = (mu_ * velocity_) - (lr_ * dw);
    return weight + velocity_.transpose();
}
