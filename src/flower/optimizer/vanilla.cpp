#include <flower/optimizer/vanilla.h>

using namespace flower;

optimizer_ptr VanillaDef::instance_;

VanillaDef::VanillaDef(double lr)
    : lr_(lr)
{}

optimizer_ptr VanillaDef::create(Net *net) const
{
    if (!instance_)
        instance_ = std::make_shared<Vanilla>(net, *this);

    return instance_;
}

Vanilla::Vanilla(Net *net, const VanillaDef &definition)
    : IOptimizer(net, definition), lr_(definition.lr())
{}

Eigen::MatrixXd Vanilla::optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &derivative)
{
    return weight - (lr_ * derivative);
}

Eigen::Tensor<double, 2> Vanilla::optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative)
{
    return weight - (lr_ * derivative);
}
