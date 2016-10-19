template <typename Scalar>
Vanilla<Scalar>::Vanilla(double lr)
    : lr_(lr)
{}

template <typename Scalar>
OptimizerPtr<Scalar> Vanilla<Scalar>::create(Net<Scalar> *net) const
{
//    if (!instance_)
//        instance_ = std::make_shared<VanillaOp<Scalar>>(net, *this);

//    return instance_;
    return std::make_shared<VanillaOp<Scalar>>(net, *this);
}

template <typename Scalar>
VanillaOp<Scalar>::VanillaOp(Net<Scalar> *net, const Vanilla<Scalar> &definition)
    : IOptimizerOp<Scalar>(net, definition), lr_(definition.lr())
{}

template <typename Scalar>
Tensor<Scalar, 2> VanillaOp<Scalar>::optimize(const Tensor<Scalar, 2> &weight, const Tensor<Scalar, 2> &derivative)
{
    return compute<2>(weight, derivative);
}

template <typename Scalar>
Tensor<Scalar, 4> VanillaOp<Scalar>::optimize(const Tensor<Scalar, 4> &weight, const Tensor<Scalar, 4> &derivative)
{
    return compute<4>(weight, derivative);
}

template <typename Scalar>
template<int rank>
Tensor<Scalar, rank> VanillaOp<Scalar>::compute(const Tensor<Scalar, rank> &weight, const Tensor<Scalar, rank> &derivative)
{
    return weight - (lr_ * derivative);
}
