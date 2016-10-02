#include <flower/layer/relu.h>
#include <cmath>

using namespace flower;

template<typename Scalar>
struct ReluOp {
  ReluOp()
  {}

  const Scalar operator()(const Scalar& x) const
  {
      return x >= 0 ? x : 0;
  }
};

template<typename Scalar>
struct ReluDerivativeOp {
    ReluDerivativeOp()
  {}

  const Scalar operator()(const Scalar& x) const
  {
      return x < 0 ? 0 : 1;
  }
};

Relu::Relu()
    : ILayerDef()
{}

layer_ptr Relu::create(Net *net, const char *name) const
{
    return std::make_shared<ReluLayer>(net, name, *this);
}

ReluLayer::ReluLayer(Net *net, const char *name, const Relu &definition)
    : ILayer(net, name, definition), data_(0, 0)
{}

Eigen::Tensor<double, 2> ReluLayer::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(ReluOp<double>());
}

Eigen::Tensor<double, 2> ReluLayer::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(ReluDerivativeOp<double>()) * errors;
}
