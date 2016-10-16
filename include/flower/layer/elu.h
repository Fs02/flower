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

        TensorData<Scalar> forward(const TensorData<Scalar> &bottom, bool train = false);
        TensorData<Scalar> backward(const TensorData<Scalar> &top);

    protected:
        Eigen::Tensor<double, 2> data_;
    };

    #include <flower/elu.inl>
}

#endif
