#ifndef FLOWER_RELU_H
#define FLOWER_RELU_H

#include <flower/layer.h>

namespace flower
{
    template<typename Scalar>
    class Relu : public ILayer<Scalar>
    {
    public:
        Relu();

        inline const char *type() const { return "Relu"; }

    protected:
        LayerPtr<Scalar> create(Net<Scalar> *net) const;
    };

    template<typename Scalar>
    class ReluOp : public ILayerOp<Scalar>
    {
    public:
        explicit ReluOp(Net<Scalar> *net, const Relu<Scalar> &definition);

        inline const char *type() const { return "Relu"; }

        Tensor<Scalar, 2, RowMajor> forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train = false);
        Tensor<Scalar, 2, RowMajor> backward(const Tensor<Scalar, 2, RowMajor> &top);

    protected:
        Tensor<Scalar, 2, RowMajor> data_;
        Relu<Scalar> relu_;
    };

    #include <flower/layer/relu.inl>
}

#endif
