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

EluDef::EluDef(double alpha)
    : ILayerDef(), alpha_(alpha)
{}

layer_ptr EluDef::create(Net *net, const char *name) const
{
    return std::make_shared<Elu>(net, name, *this);
}

Elu::Elu(Net *net, const char *name, const EluDef &definition)
    : ILayer(net, name, definition), data_(0, 0), input_(0, 0), alpha_(definition.alpha())
{}

Eigen::MatrixXd Elu::forward(const Eigen::MatrixXd &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(EluOp<double>(alpha_));
}

Eigen::MatrixXd Elu::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(EluDerivativeOp<double>(alpha_)).transpose().cwiseProduct(errors);
}


Eigen::Tensor<double, 2> Elu::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        input_ = data;

    return data.unaryExpr(EluOp<double>(alpha_));
}

Eigen::Tensor<double, 2> Elu::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return input_.shuffle(transpose).unaryExpr(EluDerivativeOp<double>(alpha_)) * errors;
}
