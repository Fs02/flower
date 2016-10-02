#include <flower/layer/relu.h>
#include <cmath>

using namespace flower;

inline double relu(double x) { return x >= 0 ? x : 0; }
inline double d_relu(double x) { return x < 0 ? 0 : 1; }

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

ReluDef::ReluDef()
    : ILayerDef()
{}

layer_ptr ReluDef::create(Net *net, const char *name) const
{
    return std::make_shared<Relu>(net, name, *this);
}

Relu::Relu(Net *net, const char *name, const ReluDef &definition)
    : ILayer(net, name, definition), data_(0, 0), input_(0, 0)
{}

Eigen::MatrixXd Relu::forward(const Eigen::MatrixXd &data, bool train)
{
    if (train)
        data_ = data;

    return data.unaryExpr(&relu);
}

Eigen::MatrixXd Relu::backward(const Eigen::MatrixXd &errors)
{
    return data_.unaryExpr(&d_relu).transpose().cwiseProduct(errors);
}


Eigen::Tensor<double, 2> Relu::forward(const Eigen::Tensor<double, 2> &data, bool train)
{
    if (train)
        input_ = data;

    return data.unaryExpr(ReluOp<double>());
}

Eigen::Tensor<double, 2> Relu::backward(const Eigen::Tensor<double, 2> &errors)
{
    Eigen::array<int, 2> transpose({1, 0});

    return input_.shuffle(transpose).unaryExpr(ReluDerivativeOp<double>()) * errors;
}
