#include <flower/supervised_learning.h>
#include <flower/net.h>

using namespace flower;

SupervisedLearning::SupervisedLearning(Net *net, const IOptimizerDef &optimizer_def)
    : net_(net)
{
    // configure all layer
    for(const auto &layer : net_->layers())
    {
        layer.second->configure(optimizer_def);
    }
}

SupervisedLearning::~SupervisedLearning()
{}

double SupervisedLearning::feed(const Eigen::MatrixXd &data, const Eigen::MatrixXd &target)
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
