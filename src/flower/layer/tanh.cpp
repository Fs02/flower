#include <flower/layer/tanh.h>

using namespace flower;

TanhDef::TanhDef(unsigned int size)
    : ILayerDef(), size_(size)
{}

layer_ptr TanhDef::create(Net *net, const char *name) const
{
    return std::make_shared<Tanh>(net, name, *this);
}


Tanh::Tanh(Net* net, const char *name, const TanhDef &definition)
    : ILayer(net, name, definition), data_(0, 0)
{}

Eigen::MatrixXd Tanh::forward(const Eigen::MatrixXd &data)
{
    data_ = data;
    return data.unaryExpr(&tanh);
}

Eigen::MatrixXd Tanh::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr([](double x) { return 1.0 - pow(tanh(x), 2.0); }).transpose().cwiseProduct(errors);
}
