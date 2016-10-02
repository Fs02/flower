#include <flower/layer/dropout.h>
#include <cmath>

using namespace flower;

Dropout::Dropout(double probabilty)
    : ILayerDef(), probability_(probabilty)
{}

layer_ptr Dropout::create(Net *net, const char *name) const
{
    return std::make_shared<DropoutLayer>(net, name, *this);
}

DropoutLayer::DropoutLayer(Net *net, const char *name, const Dropout &definition)
    : ILayer(net, name, definition), mask_(0, 0), probability_(definition.probability())
{}

Eigen::Tensor<double, 2> DropoutLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
    {
        mask_ = (data.random() > probability_).cast<double>()  / probability_;

        // dropout with inverted approach
        return data * mask_;
    }

    return data;
}

Eigen::Tensor<double, 2> DropoutLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return mask_.shuffle(transpose) * errors;
}

