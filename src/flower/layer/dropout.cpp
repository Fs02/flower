#include <flower/layer/dropout.h>
#include <cmath>

using namespace flower;

DropoutDef::DropoutDef(double probabilty)
    : ILayerDef(), probability_(probabilty)
{}

layer_ptr DropoutDef::create(Net *net, const char *name) const
{
    return std::make_shared<Dropout>(net, name, *this);
}

Dropout::Dropout(Net *net, const char *name, const DropoutDef &definition)
    : ILayer(net, name, definition), mask_(0, 0), probability_(definition.probability())
{}

Eigen::MatrixXd Dropout::forward(const Eigen::MatrixXd &data)
{
    mask_ = (Eigen::ArrayXXd::Random(data.rows(), data.cols()) > probability_).cast<double>();

    // dropout with inverted approach
    return data.cwiseProduct(mask_.matrix()) / probability_;
}

Eigen::MatrixXd Dropout::backward(const Eigen::MatrixXd &errors)
{
    return mask_.matrix().transpose().cwiseProduct(errors);
}
