#include <flower/optimizer/adam.h>
#include <flower/net.h>
#include <iostream>

using namespace flower;

AdamDef::AdamDef(double lr, double beta1, double beta2, double eps)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps)
{}

optimizer_ptr AdamDef::create(Net *net) const
{
    return std::make_shared<Adam>(net, *this);
}

Adam::Adam(Net *net, const AdamDef &definition)
    : IOptimizer(net, definition), lr_(definition.lr()), beta1_(definition.beta1()), beta2_(definition.beta2()),
      eps_(definition.eps()), m_(0, 0), v_(0, 0)
{}

Eigen::MatrixXd Adam::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw)
{
    if (m_.rows() * m_.cols() == 0)
    {
        m_= Eigen::ArrayXXd(dw.rows(), dw.cols());
        v_= Eigen::ArrayXXd(dw.rows(), dw.cols());
    }

    // update first moment
    m_ = beta1_ * m_ + (1.0 - beta1_) * dw.array();
    // update second moment
    v_ = beta2_ * v_ + (1.0 - beta2_) * dw.array().pow(2.0);

    // bias correction
    m_ = m_ / (1.0 - pow(beta1_, net_->epoch()));
    v_ = v_ / (1.0 - pow(beta2_, net_->epoch()));

    return weight - (lr_ * m_ / (v_.sqrt() + eps_)).matrix().transpose();
}
