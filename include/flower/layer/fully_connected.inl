template<typename Scalar>
FullyConnected<Scalar>::FullyConnected()
    : ILayer<Scalar>(), input_size_(input_size), output_size_(output_size)
{}

template<typename Scalar>
LayerPtr<Scalar> FullyConnected::create(Net<Scalar> *net) const
{
    return std::make_shared<FullyConnectedOp<Scalar>>(net, *this);
}

template<typename Scalar>
FullyConnectedOp<Scalar>::FullyConnectedOp(Net<Scalar> *net, const FullyConnected<Scalar> &definition)
    : ILayerOp<Scalar>(net, definition), data_(0, 0),
      weights_(definition.input_size() + 1, definition.output_size()) // with bias
{
    weights_.setRandom();
}

template<typename Scalar>
void FullyConnectedLayer<Scalar>::configure(const IOptimizer<Scalar> &optimizer)
{
    optimizer_ = optimizer.create(net_);
}

template<typename Scalar>
TensorData<Scalar> FullyConnectedOp<Scalar>::forward(const TensorData<Scalar> &bottom, bool train = false)
{
    // append bias and store input
    array<std::pair<int, int>, 2> bias({std::make_pair(0, 0), std::make_pair(0, 1)});
    data_ = bottom.map<2>().pad(bias, 1);

    // compute input * weight matrix product
    array<IndexPair<int>, 1> product_dims = { IndexPair<int>(1, 0) };
    auto output = data_.contract(weights_, product_dims);

    return output;
}

template<typename Scalar>
TensorData<Scalar> FullyConnectedOp<Scalar>::backward(const TensorData<Scalar> &top)
{
    array<IndexPair<int>, 1> transpose_product_dims = { IndexPair<int>(0, 1) };
    weights_ = optimizer_->optimize(weights_, data_.contract(top.map<2>(), transpose_product_dims));

    // remove bias from weight
    array<int, 2> offsets = {0, 0};
    array<int, 2> extents = {(int)weights_.dimension(0) - 1, (int)weights_.dimension(1)};
    Tensor<double, 2> weights_nobias = weights_.slice(offsets, extents);

    // compute weight_nobias * errors matrix product
    array<IndexPair<int>, 1> product_dims = { IndexPair<int>(1, 0) };
    auto result = weights_nobias.contract(errors, product_dims);

    return result;
}
