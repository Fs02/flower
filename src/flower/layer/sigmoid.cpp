#include <flower/layer/sigmoid.h>

using namespace flower;

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double d_sigmoid(double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }

template<typename Scalar>
struct SigmoidOp {
  SigmoidOp()
  {}

  const Scalar operator()(const Scalar& x) const
  {
      return sigmoid(x);
  }
};

template<typename Scalar>
struct SigmoidDerivativeOp {
  SigmoidDerivativeOp()
  {}

  const Scalar operator()(const Scalar& x) const
  {
      return sigmoid(x) * (1.0 - sigmoid(x));
  }
};

SigmoidDef::SigmoidDef()
    : ILayerDef()
{}

layer_ptr SigmoidDef::create(Net *net, const char *name) const
{
    return std::make_shared<Sigmoid>(net, name, *this);
}


Sigmoid::Sigmoid(Net *net, const char *name, const SigmoidDef &definition)
    : ILayer(net, name, definition), data_(0, 0), input_(0, 0)
{}

Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(&sigmoid);
}

Eigen::MatrixXd Sigmoid::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(&d_sigmoid).transpose().cwiseProduct(errors);
}


Eigen::Tensor<double, 2> Sigmoid::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        input_ = data;

    return data.unaryExpr(SigmoidOp<double>());
}

Eigen::Tensor<double, 2> Sigmoid::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return input_.shuffle(transpose).unaryExpr(SigmoidDerivativeOp<double>()) * errors;
}
