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
    : ILayerOp<Scalar>(net, definition), data_(0)
{}

template<typename Scalar>
TensorData<Scalar> TanhOp<Scalar>::forward(TensorData<Scalar> &bottom, bool train)
{
    auto bottom_tensor = bottom.template tensor<1>(bottom.size());

    if (train)
        data_ = bottom_tensor;

    Tensor<Scalar, 1> result = bottom_tensor.unaryExpr(internal::TanForwardhOp<Scalar>());
    return TensorData<Scalar>(result.size(), result.data());
}

template<typename Scalar>
TensorData<Scalar> TanhOp<Scalar>::backward(TensorData<Scalar> &top)
{
    auto top_tensor = top.template map<1>(top.size());

    Tensor<Scalar, 1> result = data_.unaryExpr(internal::TanhBackwardOp<Scalar>()) * top_tensor;
    return TensorData<Scalar>(result.size(), result.data());
}
