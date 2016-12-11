#ifndef FLOWER_TANH_H
#define FLOWER_TANH_H

#include <flower/layer.h>

namespace flower
{
    template<typename Scalar>
    class Tanh : public ILayer<Scalar>
    {
    public:
        Tanh();

        inline const char *type() const { return "Tanh"; }

    protected:
        LayerPtr<Scalar> create(Net<Scalar> *net) const;
    };

    template<typename Scalar>
    class TanhOp : public ILayerOp<Scalar>
    {
    public:
        explicit TanhOp(Net<Scalar> *net, const Tanh<Scalar> &definition);

        inline const char *type() const { return "Tanh"; }

        Tensor<Scalar, 2, RowMajor> forward(const Tensor<Scalar, 2, RowMajor> &bottom, bool train = false);
        Tensor<Scalar, 2, RowMajor> backward(const Tensor<Scalar, 2, RowMajor> &top);

    protected:
        Tensor<Scalar, 2, RowMajor> data_;
        Tanh<Scalar> definition_;
    };

    #include <flower/layer/tanh.inl>
}

#endif
