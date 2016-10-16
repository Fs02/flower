#ifndef FLOWER_SIGMOID_H
#define FLOWER_SIGMOID_H

#include <flower/layer.h>

namespace flower
{
    template<typename Scalar>
    class Sigmoid : public ILayer<Scalar>
    {
    public:
        Sigmoid();

        inline const char *type() const { return "Sigmoid"; }

    protected:
        LayerPtr<Scalar> create(Net<Scalar> *net) const;
    };

    template<typename Scalar>
    class SigmoidOp : public ILayerOp<Scalar>
    {
    public:
        explicit SigmoidOp(Net<Scalar> *net, const Sigmoid<Scalar> &definition);

        TensorData<Scalar> forward(const TensorData<Scalar> &bottom, bool train = false);
        TensorData<Scalar> backward(const TensorData<Scalar> &top);

    protected:
        Tensor<Scalar, 1> data_;
    };

    #include<flower/sigmoid.inl>
}

#endif
