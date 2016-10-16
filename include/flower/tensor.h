#ifndef FLOWER_TENSOR_H
#define FLOWER_TENSOR_H

#include <Eigen/CXX11/Tensor>
#include <memory>

namespace flower
{
    using namespace Eigen;

    template<typename Scalar>
    class TensorData
    {
    public:
        explicit TensorData(unsigned int size);
        TensorData(const TensorData<Scalar> &other);

        inline std::unique_ptr<Scalar[]> &data();
        inline unsigned int size() const;

        template<int rank, typename... Indexs>
        TensorMap<Tensor<Scalar, rank>> map(Indexs... dimensions);

    private:
        std::unique_ptr<Scalar[]> data_;
        unsigned int size_;
    };

    #include <flower/tensor.inl>
}

#endif
