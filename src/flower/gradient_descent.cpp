#include <flower/gradient_descent.h>
#include <flower/optimizer/vanilla.h>
#include <flower/net.h>

using namespace flower;

GradientDescent::GradientDescent(Net *net, double learning_rate)
    : GradientDescent(net, Vanilla(learning_rate))
{}

GradientDescent::GradientDescent(Net *net, const IOptimizerDef &optimizer_def)
    : net_(net)
{
    // configure all layer
    for(const auto &layer : net_->layers())
    {
        layer->configure(optimizer_def);
    }
}

GradientDescent::~GradientDescent()
{}

Eigen::Tensor<double, 0> GradientDescent::feed(const Eigen::Tensor<double, 2> &data, const Eigen::Tensor<double, 2> &target)
{
    // increase epoch
    net_->epoch_++;

    // forward propagate
    Eigen::Tensor<double, 2> predict = data;
    for(const auto &layer : net_->layers())
    {
        predict = layer->forward(predict, true);
    }

    // TODO
    Eigen::Tensor<double, 0> total_error = (target - predict).pow(2.0).mean();

    // back propagate
    Eigen::array<int, 2> transpose({1, 0});
    Eigen::Tensor<double, 2> errors = -(target - predict).shuffle(transpose);

    for(auto i = net_->layers().rbegin(); i != net_->layers().rend(); ++i)
    {
        errors = (*i)->backward(errors);
    }

    return total_error;
}
