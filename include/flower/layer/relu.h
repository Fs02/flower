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

        TensorData<Scalar> forward(TensorData<Scalar> &bottom, bool train = false);
        TensorData<Scalar> backward(TensorData<Scalar> &top);

    protected:
        Tensor<Scalar, 1> data_;
        Relu<Scalar> relu_;
    };

    #include <flower/layer/relu.inl>
}

#endif
