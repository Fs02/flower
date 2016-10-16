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
LayerPtr<Scalar> Relu::create(Net<Scalar> *net) const
{
    return std::make_shared<ReluOp<Scalar>>(net, *this);
}


template<typename Scalar>
ReluOp<Scalar>::ReluOp(Net<Scalar> *net, const Relu<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
TensorData<Scalar> ReluOp<Scalar>::forward(const TensorData<Scalar> &bottom, bool train = false)
{
    if (train)
        data_ = bottom;

    return bottom.map<1>(bottom.size()).unaryExpr(internal::ReluForwardhOp<Scalar>());
}

template<typename Scalar>
TensorData<Scalar> ReluOp<Scalar>::backward(const TensorData<Scalar> &top)
{
    return data_.map<1>(data_.size()).unaryExpr(internal::ReluBackwardOp<Scalar>()) * top.map<1>(top.size());
}
