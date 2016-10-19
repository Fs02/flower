#ifndef FLOWER_ELU_H
#define FLOWER_ELU_H

#include <flower/layer.h>

namespace flower
{
    template<typename Scalar>
    class Elu : public ILayer<Scalar>
    {
    public:
        Elu(double alpha = 1.0);

        inline const char *type() const { return "Elu"; }

        inline double alpha() const { return alpha_; }

        double alpha_;

    protected:
        LayerPtr<Scalar> create(Net<Scalar> *net) const;
    };

    template<typename Scalar>
    class EluOp : public ILayerOp<Scalar>
    {
    public:
        explicit EluOp(Net<Scalar> *net, const Elu<Scalar> &definition);

        inline const char *type() const { return "Elu"; }

        TensorData<Scalar> forward(TensorData<Scalar> &bottom, bool train = false);
        TensorData<Scalar> backward(TensorData<Scalar> &top);

    protected:
        Tensor<Scalar, 1> data_;
        Elu<Scalar> definition_;
    };

    #include <flower/layer/elu.inl>
}

#endif
