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
Elu<Scalar>::Elu()
    : ILayer<Scalar>()
{}

template<typename Scalar>
LayerPtr<Scalar> Elu::create(Net<Scalar> *net) const
{
    return std::make_shared<EluOp<Scalar>>(net, *this);
}


template<typename Scalar>
EluOp<Scalar>::EluOp(Net<Scalar> *net, const Elu<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
TensorData<Scalar> EluOp<Scalar>::forward(const TensorData<Scalar> &bottom, bool train = false)
{
    if (train)
        data_ = bottom.map<1>(bottom.size());

    return bottom.map<1>(bottom.size()).unaryExpr(internal::EluForwardhOp<Scalar>(definition.alpha()));
}

template<typename Scalar>
TensorData<Scalar> EluOp<Scalar>::backward(const TensorData<Scalar> &top)
{
    return data_.unaryExpr(internal::EluBackwardOp<Scalar>(definition.alpha())) * top.map<1>(top.size());
}
