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

Eigen::MatrixXd Net::infer(const Eigen::MatrixXd &data) const
{
    // forward propagate
    Eigen::MatrixXd predict = data;
    for(const auto &layer : layers_)
    {
        predict = layer.second->forward(predict);
    }
    return predict;
}

void Net::add(const char *name, const ILayerDef &definition)
{
    layers_.push_back(std::make_pair(name, definition.create(this, name)));
}

const std::vector<std::pair<const char*, layer_ptr>> &Net::layers() const
{
    return layers_;
}
