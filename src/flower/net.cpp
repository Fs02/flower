#include <flower/net.h>

using namespace flower;

double square(double x) {
    return x * x;
}

Net::Net()
    : layers_(), epoch_(0)
{}

Net::~Net()
{
    layers_.clear();
}

Eigen::Tensor<double, 2> Net::infer(const Eigen::Tensor<double, 2> &data) const
{
    // forward propagate
    Eigen::Tensor<double, 2> predict = data;
    for(const auto &layer : layers_)
    {
        predict = layer->forward(predict);
    }
    return predict;
}

void Net::add(const ILayerDef &definition)
{
    layers_.push_back(definition.create(this));
}

const std::vector<layer_ptr> &Net::layers() const
{
    return layers_;
}
