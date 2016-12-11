template<typename Scalar>
Convolution<Scalar>::Convolution(const array<int, 3>& filter_dims, int filter_num, int stride, int padding)
    : ILayer(), filter_dims_(filter_dims), filter_num_(filter_num), stride_(stride), padding_(padding)
{}

template<typename Scalar>
LayerPtr<Scalar> Convolution::create(Net *net) const
{
    return std::make_shared<ConvolutionOp<Scalar>>(net, *this);
}

template<typename Scalar>
ConvolutionOp<Scalar>::ConvolutionOp(Net<Scalar> *net, const Convolution &definition)
    : ILayer(net, definition), filters_(), biases_(),
      stride_(definition.stride()), padding_(definition.padding())
{
    weights_.setRandom();
    biases_.setZero();
}

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

template<typename Scalar>
void ConvolutionOp<Scalar>::convolve(const Eigen::Tensor<Scalar, 4>& input, Eigen::Tensor<Scalar, 3>& output, const Eigen::Tensor<Scalar, 3>& filter, Scalar bias, int stride)
{
    // TODO: replace with eigen built in convolve
    for (int i = 0; i < input.dimension(0); ++i)
    {
        for (int j = 0; j <= input.dimension(2) - filter.dimension(1); j += stride)
        {
            for (int k = 0; k <= input.dimension(3) - filter.dimension(2); k += stride)
            {
                // perform dot product between local region and kernel
                Eigen::array<long int, 4> offsets = {i, 0, j, k};
                Eigen::array<long int, 4> extents = {1, filter.dimension(0), filter.dimension(1), filter.dimension(2)};
                Eigen::Tensor<double, 3> local = input.slice(offsets, extents).reshape(filter.dimensions());
                Eigen::Tensor<double, 0> dot_product = (local * filter).sum();
                output(i, j/stride, k/stride) = dot_product(0) + bias;
            }
        }
    }
}
