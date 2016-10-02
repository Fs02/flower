#include <flower/layer/elu.h>
#include <cmath>

using namespace flower;

template<typename Scalar>
struct EluOp {
  EluOp(const Scalar& alpha)
      : alpha_(alpha)
  {}

  const Scalar operator()(const Scalar& x) const
  {
      // x for x >= 0
      // alpha * (exp(x) - 1.0) otherwise
      return x >= 0 ? x : alpha_ * (exp(x) - 1.0);
  }

  Scalar alpha_;
};

template<typename Scalar>
struct EluDerivativeOp {
  EluDerivativeOp(const Scalar& alpha)
      : alpha_(alpha)
  {}

  const Scalar operator()(const Scalar& x) const
  {
      // 1 for x >= 0
      // alpha * exp(x) otherwise
      return x >= 0 ? 1 : alpha_ * exp(x);
  }

  Scalar alpha_;
};

Elu::Elu(double alpha)
    : ILayerDef(), alpha_(alpha)
{}

layer_ptr Elu::create(Net *net) const
{
    return std::make_shared<EluLayer>(net, *this);
}

EluLayer::EluLayer(Net *net, const Elu &definition)
    : ILayer(net, definition), data_(0, 0), alpha_(definition.alpha())
{}

Eigen::Tensor<double, 2> EluLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(EluOp<double>(alpha_));
}

Eigen::Tensor<double, 2> EluLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(EluDerivativeOp<double>(alpha_)) * errors;
}
