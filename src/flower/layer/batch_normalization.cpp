#include <flower/layer/batch_normalization.h>
#include <cmath>

using namespace flower;

BatchNormalizationDef::BatchNormalizationDef()
    : ILayerDef()
{}

layer_ptr BatchNormalizationDef::create(Net *net) const
{
    return std::make_shared<BatchNormalization>(net, *this);
}

BatchNormalization::BatchNormalization(Net *net, const BatchNormalizationDef &definition)
    : ILayer(net, definition)
{}

Eigen::Tensor<double, 2> BatchNormalization::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    return data;
}

Eigen::Tensor<double, 2> BatchNormalization::backward(const Eigen::Tensor<double, 2> &errors)
{
    return errors;
}

