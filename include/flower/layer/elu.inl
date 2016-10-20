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
    : ILayerOp<Scalar>(net, definition), data_(0)
{}

template<typename Scalar>
TensorData<Scalar> EluOp<Scalar>::forward(TensorData<Scalar> &bottom, bool train)
{
    auto bottom_tensor = bottom.template map<1>(bottom.size());

    if (train)
        data_ = bottom_tensor;

    Tensor<Scalar, 1> result = bottom_tensor.unaryExpr(internal::EluForwardhOp<Scalar>(definition_.alpha()));
    return TensorData<Scalar>(result.data(), result.size());
}

template<typename Scalar>
TensorData<Scalar> EluOp<Scalar>::backward(TensorData<Scalar> &top)
{
    auto top_tensor = top.template map<1>(top.size());

    Tensor<Scalar, 1> result = data_.unaryExpr(internal::EluBackwardOp<Scalar>(definition_.alpha())) * top_tensor;
    return TensorData<Scalar>(result.data(), result.size());
}
