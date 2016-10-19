template<typename Scalar>
TensorData<Scalar>::TensorData(unsigned int size, Scalar *data)
    : data_(nullptr), size_(size)
{
    data_ = new Scalar[size];
    std::copy(data, data + size, data_);
}

template<typename Scalar>
TensorData<Scalar>::~TensorData()
{
    delete [] data_;
}

template<typename Scalar>
Scalar *TensorData<Scalar>::data()
{
    return data_;
}

template<typename Scalar>
unsigned int TensorData<Scalar>::size() const
{
    return size_;
}

template<typename Scalar>
template<int rank, typename... Indexs>
TensorMap<Tensor<Scalar, rank>> TensorData<Scalar>::map(Indexs... dimensions)
{
    return TensorMap<Tensor<Scalar, rank>>(data_, dimensions...);
}

template<typename Scalar>
template<int rank, typename... Indexs>
Tensor<Scalar, rank> TensorData<Scalar>::tensor(Indexs... dimensions) const
{
    Tensor<Scalar, rank> t(dimensions...);
    std::copy(data_, data_ + size_, t.data());
    return t;
}
