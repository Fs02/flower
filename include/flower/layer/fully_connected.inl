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
//    optimizer_ = optimizer.create(net_);
}

template<typename Scalar>
TensorData<Scalar> FullyConnectedOp<Scalar>::forward(TensorData<Scalar> &bottom, bool train)
{
    auto bottom_tensor = bottom.template map<2>(bottom.size()/definition_.input_size(), definition_.input_size());

    // append bias and store input
    array<std::pair<int, int>, 2> bias({std::make_pair(0, 0), std::make_pair(0, 1)});
    data_ = bottom_tensor.pad(bias, 1);

    // compute input * weight matrix product
    array<IndexPair<int>, 1> product_dims = { IndexPair<int>(1, 0) };
    Tensor<Scalar, 2> result = bottom_tensor.pad(bias, 1).contract(weights_, product_dims);

    return TensorData<Scalar>(result.data(), result.size());
}

template<typename Scalar>
TensorData<Scalar> FullyConnectedOp<Scalar>::backward(TensorData<Scalar> &top)
{
    array<int, 2> transpose({1, 0});
    auto top_tensor = top.template map<2>(top.size()/definition_.output_size(), definition_.output_size()).shuffle(transpose);

    array<IndexPair<int>, 1> transpose_product_dims = { IndexPair<int>(0, 1) };
//    weights_ = optimizer_->optimize(weights_, data_.contract(top_tensor, transpose_product_dims));
    weights_ = weights_ - (0.01 * data_.contract(top_tensor, transpose_product_dims));

    // remove bias from weight
    array<int, 2> offsets = {0, 0};
    array<int, 2> extents = {(int)weights_.dimension(0) - 1, (int)weights_.dimension(1)};
    Tensor<Scalar, 2> weights_nobias = weights_.slice(offsets, extents);

    // compute weight_nobias * errors matrix product
    array<IndexPair<int>, 1> product_dims = { IndexPair<int>(1, 0) };
    Tensor<Scalar, 2> result = weights_nobias.contract(top_tensor, product_dims);

    return TensorData<Scalar>(result.data(), result.size());
}
