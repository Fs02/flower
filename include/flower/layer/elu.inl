namespace internal {
    template<typename Scalar>
    struct EluForwardhOp {
        EluForwardhOp(const Scalar& alpha)
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
    struct EluBackwardOp {
        EluBackwardOp(const Scalar& alpha)
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
}

template<typename Scalar>
Elu<Scalar>::Elu(double alpha)
    : ILayer<Scalar>(), alpha_(alpha)
{}

template<typename Scalar>
LayerPtr<Scalar> Elu<Scalar>::create(Net<Scalar> *net) const
{
    return std::make_shared<EluOp<Scalar>>(net, *this);
}


template<typename Scalar>
EluOp<Scalar>::EluOp(Net<Scalar> *net, const Elu<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> EluOp<Scalar>::forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train)
{
    if (train)
        data_ = bottom;

    return bottom.unaryExpr(internal::EluForwardhOp<Scalar>(definition_.alpha()));
}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> EluOp<Scalar>::backward(const Tensor<Scalar, 2, RowMajor> &top)
{
    Eigen::array<int, 2> transpose({1, 0});

    return data_.shuffle(transpose).unaryExpr(internal::EluBackwardOp<Scalar>(definition_.alpha())) * top;
}
