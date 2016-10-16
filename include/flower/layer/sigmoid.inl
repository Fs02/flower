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
LayerPtr<Scalar> Sigmoid::create(Net<Scalar> *net) const
{
    return std::make_shared<SigmoidOp<Scalar>>(net, *this);
}


template<typename Scalar>
SigmoidOp<Scalar>::SigmoidOp(Net<Scalar> *net, const Sigmoid<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0)
{}

template<typename Scalar>
TensorData<Scalar> SigmoidOp<Scalar>::forward(const TensorData<Scalar> &bottom, bool train = false)
{
    if (train)
        data_ = bottom.map<1>(bottom.size());

    return bottom.map<1>(bottom.size()).unaryExpr(internal::SigmoidForwardhOp<Scalar>());
}

template<typename Scalar>
TensorData<Scalar> SigmoidOp<Scalar>::backward(const TensorData<Scalar> &top)
{
    return data_.unaryExpr(internal::SigmoidBackwardOp<Scalar>()) * top.map<1>(top.size());
}
