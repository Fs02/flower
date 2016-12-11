namespace internal {
    inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

    template<typename Scalar>
    struct SigmoidForwardhOp {
        SigmoidForwardhOp()
        {}

        const Scalar operator()(const Scalar& x) const
        {
            return sigmoid(x);
        }
    };

    template<typename Scalar>
    struct SigmoidBackwardOp {
        SigmoidBackwardOp()
        {}

        const Scalar operator()(const Scalar& x) const
        {
            return sigmoid(x) * (1.0 - sigmoid(x));
        }
    };
}

template<typename Scalar>
Sigmoid<Scalar>::Sigmoid()
    : ILayer<Scalar>()
{}

template<typename Scalar>
LayerPtr<Scalar> Sigmoid<Scalar>::create(Net<Scalar> *net) const
{
    return std::make_shared<SigmoidOp<Scalar>>(net, *this);
}

template<typename Scalar>
SigmoidOp<Scalar>::SigmoidOp(Net<Scalar> *net, const Sigmoid<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> SigmoidOp<Scalar>::forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train)
{
    if (train)
        data_ = bottom;

    return bottom.unaryExpr(internal::SigmoidForwardhOp<Scalar>());
}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> SigmoidOp<Scalar>::backward(const Tensor<Scalar, 2, RowMajor> &top)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(internal::SigmoidBackwardOp<Scalar>()) * top;
}
