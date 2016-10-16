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
LayerPtr<Scalar> Tanh::create(Net<Scalar> *net) const
{
    return std::make_shared<TanhOp<Scalar>>(net, *this);
}


template<typename Scalar>
TanhOp<Scalar>::TanhOp(Net<Scalar> *net, const Tanh<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
TensorData<Scalar> TanhOp<Scalar>::forward(const TensorData<Scalar> &bottom, bool train = false)
{
    if (train)
        data_ = bottom;

    return bottom.map<1>(bottom.size()).unaryExpr(internal::TanForwardhOp<Scalar>());
}

template<typename Scalar>
TensorData<Scalar> TanhOp<Scalar>::backward(const TensorData<Scalar> &top)
{
    return data_.map<1>(data_.size()).unaryExpr(internal::TanhBackwardOp<Scalar>()) * top.map<1>(top.size());
}
