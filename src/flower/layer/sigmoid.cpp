#include <flower/layer/sigmoid.h>

using namespace flower;

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

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

Sigmoid::Sigmoid()
    : ILayerDef()
{}

layer_ptr Sigmoid::create(Net *net) const
{
    return std::make_shared<SigmoidLayer>(net, *this);
}


SigmoidLayer::SigmoidLayer(Net *net, const Sigmoid &definition)
    : ILayer(net, definition), data_(0, 0)
{}

Eigen::Tensor<double, 2> SigmoidLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(SigmoidOp<double>());
}

Eigen::Tensor<double, 2> SigmoidLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(SigmoidDerivativeOp<double>()) * errors;
}
