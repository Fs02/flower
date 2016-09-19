#include <flower/layer/relu.h>
#include <cmath>

using namespace flower;

inline double relu(double x) { return x >= 0 ? x : 0; }
inline double d_relu(double x) { return x < 0 ? 0 : 1; }

ReluDef::ReluDef()
    : ILayerDef()
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
    return data.unaryExpr(&relu);
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(&d_relu).transpose().cwiseProduct(errors);
}
