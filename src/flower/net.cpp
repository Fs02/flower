#include <flower/net.h>

using namespace flower;

double square(double x) {
    return x * x;
}

Net::Net()
    : layers_()
{}

Net::~Net()
{
    layers_.clear();
}

void Net::configure(const IOptimizerDef &optimizer_def)
{
    for(const auto &layer : layers_)
    {
        layer.second->configure(optimizer_def);
    }
}


double Net::train(const Eigen::MatrixXd &data, const Eigen::MatrixXd &target)
{
    // forward propagate
    Eigen::MatrixXd predict = data;
    for(const auto &layer : layers_)
    {
        predict = layer.second->forward(predict);
    }

    auto total_error = (target - predict).unaryExpr(&square).sum() / target.cols();

    // back propagate
    Eigen::MatrixXd errors = -(target - predict).transpose();
    for(auto i = layers_.rbegin(); i != layers_.rend(); ++i)
    {
        errors = (*i).second->backward(errors);
    }

    return total_error;
}

double Net::eval()
{
    return 0.0;
}

void Net::add(const char *name, const ILayerDef &definition)
{
    layers_.push_back(std::make_pair(name, definition.create(this, name)));
}
