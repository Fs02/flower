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
    : ILayer(net, name, definition), data_(0, 0), alpha_(definition.alpha())
{}

Eigen::MatrixXd Elu::forward(const Eigen::MatrixXd &data)
{
    data_ = data;
    return data.unaryExpr(EluOp<double>(alpha_));
}

Eigen::MatrixXd Elu::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(EluDerivativeOp<double>(alpha_)).transpose().cwiseProduct(errors);
}
