template<typename Scalar>
TensorData<Scalar>::TensorData(unsigned int size)
    : data_{new Scalar[size]}, size_(size)
{}

template<typename Scalar>
TensorData(const TensorData<Scalar> &other)
    : data_{new Scalar(*other.data_)}, size_(other.size)
{}


template<typename Scalar>
std::unique_ptr<Scalar[]> &TensorData<Scalar>::data()
{
    return data_;
}

template<typename Scalar>
unsigned int TensorData<Scalar>::size() const
{
    return data;
}

template<typename Scalar>
template<int rank, typename... Indexs>
TensorMap<Tensor<Scalar, rank>> TensorData<Scalar>::map(Indexs... dimensions)
{
    Scalar *storage = data_.get();
    return TensorMap<Tensor<Scalar, rank>>(storage, dimensions...);
}
