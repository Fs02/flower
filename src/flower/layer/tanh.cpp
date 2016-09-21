#include <flower/layer/tanh.h>

using namespace flower;

inline double d_tanh(double x) { return 1.0 - pow(tanh(x), 2.0); }

TanhDef::TanhDef()
    : ILayerDef()
{}

layer_ptr TanhDef::create(Net *net, const char *name) const
{
    return std::make_shared<Tanh>(net, name, *this);
}


Tanh::Tanh(Net *net, const char *name, const TanhDef &definition)
    : ILayer(net, name, definition), data_(0, 0)
{}

Eigen::MatrixXd Tanh::forward(const Eigen::MatrixXd &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(&tanh);
}

Eigen::MatrixXd Tanh::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(&d_tanh).transpose().cwiseProduct(errors);
}
