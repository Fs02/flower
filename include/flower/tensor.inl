template<typename Scalar>
TensorData<Scalar>::TensorData(Scalar *data, std::size_t size)
    : data_(new Scalar[size]), size_(size)
{
    std::copy(data, data + size, data_);
}

template<typename Scalar>
TensorData<Scalar>::TensorData(const TensorData& other)
    : data_(new Scalar[other.size_]), size_(other.size_)
{
    std::copy(other.data_, other.data_ + other.size_, data_);
}

template<typename Scalar>
TensorData<Scalar>::~TensorData()
{
    delete [] data_;
}

template<typename Scalar>
TensorData<Scalar> &TensorData<Scalar>::operator=(const TensorData<Scalar>& other)
{
    TensorData<Scalar> tmp(other);

    std::swap(size_, tmp.size_);
    std::swap(data_, tmp.data_);

    return *this;
}

template<typename Scalar>
Scalar *TensorData<Scalar>::data()
{
    return data_;
}

template<typename Scalar>
std::size_t TensorData<Scalar>::size() const
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
