#include <flower/layer/tanh.h>

using namespace flower;

template<typename Scalar>
struct TanhOp {
    TanhOp()
    {}

    const Scalar operator()(const Scalar& x) const
    {
        return tanh(x);
    }
};

template<typename Scalar>
struct TanhDerivativeOp {
    TanhDerivativeOp()
    {}

    const Scalar operator()(const Scalar& x) const
    {
        return 1.0 - pow(tanh(x), 2.0);
    }
};

inline double d_tanh(double x) { return 1.0 - pow(tanh(x), 2.0); }

TanhDef::TanhDef()
    : ILayerDef()
{}

layer_ptr TanhDef::create(Net *net, const char *name) const
{
    return std::make_shared<Tanh>(net, name, *this);
}


Tanh::Tanh(Net *net, const char *name, const TanhDef &definition)
    : ILayer(net, name, definition), data_(0, 0), input_(0, 0)
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


Eigen::Tensor<double, 2> Tanh::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        input_ = data;

    return data.unaryExpr(TanhOp<double>());
}

Eigen::Tensor<double, 2> Tanh::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return input_.shuffle(transpose).unaryExpr(TanhDerivativeOp<double>()) * errors;
}
