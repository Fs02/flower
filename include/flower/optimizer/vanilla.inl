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
Tensor<Scalar, 2, RowMajor> VanillaOp<Scalar>::optimize(const Tensor<Scalar, 2, RowMajor> &weight, const Tensor<Scalar, 2, RowMajor> &derivative)
{
    return compute<2>(weight, derivative);
}

template <typename Scalar>
Tensor<Scalar, 4, RowMajor> VanillaOp<Scalar>::optimize(const Tensor<Scalar, 4, RowMajor> &weight, const Tensor<Scalar, 4, RowMajor> &derivative)
{
    return compute<4>(weight, derivative);
}

template <typename Scalar>
template<int rank>
Tensor<Scalar, rank, RowMajor> VanillaOp<Scalar>::compute(const Tensor<Scalar, rank, RowMajor> &weight, const Tensor<Scalar, rank, RowMajor> &derivative)
{
    return weight - (lr_ * derivative);
}
