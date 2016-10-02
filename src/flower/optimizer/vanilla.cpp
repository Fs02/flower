#include <flower/optimizer/vanilla.h>

using namespace flower;

optimizer_ptr Vanilla::instance_;

Vanilla::Vanilla(double lr)
    : lr_(lr)
{}

optimizer_ptr Vanilla::create(Net *net) const
{
    if (!instance_)
        instance_ = std::make_shared<VanillaOptimizer>(net, *this);

    return instance_;
}

VanillaOptimizer::VanillaOptimizer(Net *net, const Vanilla &definition)
    : IOptimizer(net, definition), lr_(definition.lr())
{}

Eigen::Tensor<double, 2> VanillaOptimizer::optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative)
{
    return weight - (lr_ * derivative);
}
