#include <flower/gradient_descent.h>
#include <flower/optimizer/vanilla.h>
#include <flower/net.h>

using namespace flower;

GradientDescent::GradientDescent(Net *net, double learning_rate)
    : GradientDescent(net, VanillaDef(learning_rate))
{}

GradientDescent::GradientDescent(Net *net, const IOptimizerDef &optimizer_def)
    : net_(net)
{
    // configure all layer
    for(const auto &layer : net_->layers())
    {
        layer.second->configure(optimizer_def);
    }
}

GradientDescent::~GradientDescent()
{}

double GradientDescent::feed(const Eigen::MatrixXd &data, const Eigen::MatrixXd &target)
{
    // increase epoch
    net_->epoch_++;

    // forward propagate
    Eigen::MatrixXd predict = data;
    for(const auto &layer : net_->layers())
    {
        predict = layer.second->forward(predict, true);
    }

    // TODO: fix
    auto total_error = (target - predict).array().pow(2.0).sum() / target.size();

    // back propagate
    Eigen::MatrixXd errors = -(target - predict).transpose();
    for(auto i = net_->layers().rbegin(); i != net_->layers().rend(); ++i)
    {
        errors = (*i).second->backward(errors);
    }

    return total_error;
}

Eigen::Tensor<double, 0> GradientDescent::feed(const Eigen::Tensor<double, 2> &data, const Eigen::Tensor<double, 2> &target)
{
    // increase epoch
    net_->epoch_++;

    // forward propagate
    Eigen::Tensor<double, 2> predict = data;
    for(const auto &layer : net_->layers())
    {
        predict = layer.second->forward(predict, true);
    }

    Eigen::Tensor<double, 0> total_error = (target - predict).pow(2.0).mean();

    // back propagate
    Eigen::array<int, 2> transpose({1, 0});
    Eigen::Tensor<double, 2> errors = -(target - predict).shuffle(transpose);

    for(auto i = net_->layers().rbegin(); i != net_->layers().rend(); ++i)
    {
        errors = (*i).second->backward(errors);
    }

    return total_error;
}
