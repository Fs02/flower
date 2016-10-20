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
    : ILayerOp<Scalar>(net, definition), data_(0)
{}

template<typename Scalar>
TensorData<Scalar> ReluOp<Scalar>::forward(TensorData<Scalar> &bottom, bool train)
{
    auto bottom_tensor = bottom.template map<1>(bottom.size());

    if (train)
        data_ = bottom_tensor;

    Tensor<Scalar, 1> result = bottom_tensor.unaryExpr(internal::ReluForwardhOp<Scalar>());
    return TensorData<Scalar>(result.data(), result.size());
}

template<typename Scalar>
TensorData<Scalar> ReluOp<Scalar>::backward(TensorData<Scalar> &top)
{
    auto top_tensor = top.template map<1>(top.size());

    Tensor<Scalar, 1> result = data_.unaryExpr(internal::ReluBackwardOp<Scalar>()) * top_tensor;
    return TensorData<Scalar>(result.data(), result.size());
}
