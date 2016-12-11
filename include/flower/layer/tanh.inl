namespace internal {
    template<typename Scalar>
    struct TanForwardhOp {
        TanForwardhOp()
        {}

        const Scalar operator()(const Scalar& x) const
        {
            return tanh(x);
        }
    };

    template<typename Scalar>
    struct TanhBackwardOp {
        TanhBackwardOp()
        {}

        const Scalar operator()(const Scalar& x) const
        {
            return 1.0 - pow(tanh(x), 2.0);
        }
    };
}

template<typename Scalar>
Tanh<Scalar>::Tanh()
    : ILayer<Scalar>()
{}

template<typename Scalar>
LayerPtr<Scalar> Tanh<Scalar>::create(Net<Scalar> *net) const
{
    return std::make_shared<TanhOp<Scalar>>(net, *this);
}


template<typename Scalar>
TanhOp<Scalar>::TanhOp(Net<Scalar> *net, const Tanh<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> TanhOp<Scalar>::forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train)
{
    if (train)
        data_ = bottom;

    return bottom_tensor.unaryExpr(internal::TanForwardhOp<Scalar>());
}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> TanhOp<Scalar>::backward(const Tensor<Scalar, 2, RowMajor> &top)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(internal::TanhBackwardOp<Scalar>()) * top;
}
