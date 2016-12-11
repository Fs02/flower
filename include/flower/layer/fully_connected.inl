template<typename Scalar>
FullyConnected<Scalar>::FullyConnected(unsigned int input_size, unsigned int output_size)
    : ILayer<Scalar>(), input_size_(input_size), output_size_(output_size)
{}

template<typename Scalar>
LayerPtr<Scalar> FullyConnected<Scalar>::create(Net<Scalar> *net) const
{
    return std::make_shared<FullyConnectedOp<Scalar>>(net, *this);
}

template<typename Scalar>
FullyConnectedOp<Scalar>::FullyConnectedOp(Net<Scalar> *net, const FullyConnected<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0), definition_(definition),
      weights_(definition.input_size() + 1, definition.output_size()) // with bias
{
    weights_.setRandom();
}

template<typename Scalar>
void FullyConnectedOp<Scalar>::configure(const IOptimizer<Scalar> &optimizer)
{
    optimizer_ = optimizer.create(this->net_);
}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> FullyConnectedOp<Scalar>::forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train)
{
    // append bias and store input
    array<std::pair<int, int>, 2> bias({std::make_pair(0, 0), std::make_pair(0, 1)});
    data_ = bottom.pad(bias, 1);

    // compute input * weight matrix product
    array<IndexPair<int>, 1> product_dims = { IndexPair<int>(1, 0) };
    return bottom.pad(bias, 1).contract(weights_, product_dims);
}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> FullyConnectedOp<Scalar>::backward(const Tensor<Scalar, 2, RowMajor> &top)
{
    array<IndexPair<int>, 1> transpose_product_dims = { IndexPair<int>(0, 1) };
    weights_ = optimizer_->optimize(weights_, data_.contract(top, transpose_product_dims));

    // remove bias from weight
    array<int, 2> offsets = {0, 0};
    array<int, 2> extents = {(int)weights_.dimension(0) - 1, (int)weights_.dimension(1)};
    Tensor<Scalar, 2, RowMajor> weights_nobias = weights_.slice(offsets, extents);

    // compute weight_nobias * errors matrix product
    array<IndexPair<int>, 1> product_dims = { IndexPair<int>(1, 0) };
    return weights_nobias.contract(top, product_dims);
}

template<typename Scalar>
Tensor<Scalar, 2, RowMajor> &FullyConnectedOp<Scalar>::weights()
{
    return weights_;
}
