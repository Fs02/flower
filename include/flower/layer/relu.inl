namespace internal {
    template<typename Scalar>
    struct ReluForwardhOp {
        ReluForwardhOp()
        {}

        const Scalar operator()(const Scalar& x) const
        {
            return x >= 0 ? x : 0;
        }
    };

    template<typename Scalar>
    struct ReluBackwardOp {
        ReluBackwardOp()
        {}

        const Scalar operator()(const Scalar& x) const
        {
            return x < 0 ? 0 : 1;
        }
    };
}

template<typename Scalar>
Relu<Scalar>::Relu()
    : ILayer<Scalar>()
{}

template<typename Scalar>
LayerPtr<Scalar> Relu<Scalar>::create(Net<Scalar> *net) const
{
    return std::make_shared<ReluOp<Scalar>>(net, *this);
}

template<typename Scalar>
ReluOp<Scalar>::ReluOp(Net<Scalar> *net, const Relu<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> ReluOp<Scalar>::forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train)
{
    if (train)
        data_ = bottom;

    return bottom.unaryExpr(internal::ReluForwardhOp<Scalar>());
}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> ReluOp<Scalar>::backward(const Tensor<Scalar, 2, RowMajor> &top)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(internal::ReluBackwardOp<Scalar>()) * top;
}
