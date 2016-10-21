#ifndef FLOWER_TENSOR_H
#define FLOWER_TENSOR_H

#include <Eigen/CXX11/Tensor>
#include <memory>
#include <iterator>

namespace flower
{
    using namespace Eigen;

    template<typename Scalar>
    class TensorData
    {
    public:
        explicit TensorData(Scalar *data, std::size_t size);
        TensorData(const TensorData &other);
        TensorData(TensorData &&other);
        ~TensorData();

        TensorData<Scalar> &operator=(const TensorData<Scalar> &other);
        TensorData<Scalar> &operator=(TensorData<Scalar> &&other);

        inline Scalar *data();
        inline std::size_t size() const;

        template<int rank, typename... Indexs>
        TensorMap<Tensor<Scalar, rank>> map(Indexs... dimensions);

        template<int rank, typename... Indexs>
        Tensor<Scalar, rank> tensor(Indexs... dimensions) const;

    private:
        Scalar *data_;
        std::size_t size_;
    };

    #include <flower/tensor.inl>
}

#endif
