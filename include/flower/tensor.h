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
        explicit TensorData(unsigned int size, Scalar *data);
        ~TensorData();

        inline Scalar *data();
        inline unsigned int size() const;

        template<int rank, typename... Indexs>
        TensorMap<Tensor<Scalar, rank>> map(Indexs... dimensions);

        template<int rank, typename... Indexs>
        Tensor<Scalar, rank> tensor(Indexs... dimensions) const;

    private:
        Scalar *data_;
        unsigned int size_;
    };

    #include <flower/tensor.inl>
}

#endif
