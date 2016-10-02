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

Tanh::Tanh()
    : ILayerDef()
{}

layer_ptr Tanh::create(Net *net) const
{
    return std::make_shared<TanhLayer>(net, *this);
}


TanhLayer::TanhLayer(Net *net, const Tanh &definition)
    : ILayer(net, definition), data_(0, 0)
{}

Eigen::Tensor<double, 2> TanhLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(TanhOp<double>());
}

Eigen::Tensor<double, 2> TanhLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(TanhDerivativeOp<double>()) * errors;
}
