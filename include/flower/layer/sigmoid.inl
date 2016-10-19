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
LayerPtr<Scalar> Sigmoid<Scalar>::create(Net<Scalar> *net) const
{
    return std::make_shared<SigmoidOp<Scalar>>(net, *this);
}


template<typename Scalar>
SigmoidOp<Scalar>::SigmoidOp(Net<Scalar> *net, const Sigmoid<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0)
{}

template<typename Scalar>
TensorData<Scalar> SigmoidOp<Scalar>::forward(TensorData<Scalar> &bottom, bool train)
{
    auto bottom_tensor = bottom.template map<1>(bottom.size());

    if (train)
        data_ = bottom_tensor;

    std::cout << data_.data() << std::endl;
    std::cout << bottom_tensor.data() << std::endl;

    Tensor<Scalar, 1> result = bottom_tensor.unaryExpr(internal::SigmoidForwardhOp<Scalar>());
    return TensorData<Scalar>(result.size(), result.data());
}

template<typename Scalar>
TensorData<Scalar> SigmoidOp<Scalar>::backward(TensorData<Scalar> &top)
{
    auto top_tensor = top.template map<1>(top.size());

    Tensor<Scalar, 1> result = data_.unaryExpr(internal::SigmoidBackwardOp<Scalar>()) * top_tensor;
    return TensorData<Scalar>(result.size(), result.data());
}
