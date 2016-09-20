#include <flower/optimizer/momentum.h>

using namespace flower;

MomentumDef::MomentumDef(double learning_rate, double mu)
    : learning_rate_(learning_rate), mu_(mu)
{}

optimizer_ptr MomentumDef::create(Net *net) const
{
    return std::make_shared<Momentum>(net, *this);
}

Momentum::Momentum(Net* net, const MomentumDef &definition)
    : IOptimizer(net, definition), learning_rate_(definition.learning_rate()), mu_(definition.mu()), velocity_(0, 0)
{}

Eigen::MatrixXd Momentum::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw)
{
    if (velocity_.rows() * velocity_.cols() == 0)
        velocity_ = Eigen::MatrixXd(dw.rows(), dw.cols());

    velocity_ = (mu_ * velocity_) - (learning_rate_ * dw);
    return weight + velocity_.transpose();
}
