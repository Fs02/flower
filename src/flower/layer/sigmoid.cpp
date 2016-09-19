#include <flower/layer/sigmoid.h>

using namespace flower;

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double d_sigmoid(double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }

SigmoidDef::SigmoidDef(unsigned int size)
    : ILayerDef(), size_(size)
{}

layer_ptr SigmoidDef::create(Net *net, const char *name) const
{
    return std::make_shared<Sigmoid>(net, name, *this);
}


Sigmoid::Sigmoid(Net* net, const char *name, const SigmoidDef &definition)
    : ILayer(net, name, definition), data_(0, 0)
{}

Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd &data)
{
    data_ = data;
    return data.unaryExpr(&sigmoid);
}

Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(&d_sigmoid).transpose().cwiseProduct(errors);
}
