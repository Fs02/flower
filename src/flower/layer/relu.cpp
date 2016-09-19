#include <flower/layer/relu.h>
#include <cmath>

using namespace flower;

inline double max(double x) { return x >= 0 ? x : 0; }
inline double d_max(double x) { return x < 0 ? 0 : 1; }

ReluDef::ReluDef(unsigned int size)
    : ILayerDef(), size_(size)
{}

layer_ptr ReluDef::create(Net *net, const char *name) const
{
    return std::make_shared<Relu>(net, name, *this);
}

Relu::Relu(Net* net, const char *name, const ReluDef &definition)
    : ILayer(net, name, definition), data_(0, 0)
{}

Eigen::MatrixXd Relu::forward(const Eigen::MatrixXd &data)
{
    data_ = data;
    return data.unaryExpr(&max);
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(&d_max).transpose().cwiseProduct(errors);
}
