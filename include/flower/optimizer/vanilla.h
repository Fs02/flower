#ifndef FLOWER_VANILLA_H
#define FLOWER_VANILLA_H

#include <flower/optimizer.h>

namespace flower
{
    template<typename Scalar>
    class Vanilla : public IOptimizer<Scalar>
    {
    public:
        Vanilla(double lr = 0.01);

        inline const char *type() const { return "Vanilla"; }

        inline double lr() const { return lr_; }

    protected:
        OptimizerPtr<Scalar> create(Net<Scalar> *net) const;
        static OptimizerPtr<Scalar> instance_;

        double lr_;
    };

    template<typename Scalar>
    class VanillaOp : public IOptimizerOp<Scalar>
    {
    public:
        explicit VanillaOp(Net<Scalar> *net, const Vanilla<Scalar> &definition);

        inline const char *type() const { return "Vanilla"; }

        Tensor<Scalar, 2> optimize(const Tensor<Scalar, 2> &weight, const Tensor<Scalar, 2> &derivative);
        Tensor<Scalar, 4> optimize(const Tensor<Scalar, 4> &weight, const Tensor<Scalar, 4> &derivative);

        template<int rank>
        Tensor<Scalar, rank> compute(const Tensor<Scalar, rank> &weight, const Tensor<Scalar, rank> &derivative);

    protected:
        double lr_;
    };

    #include <flower/optimizer/vanilla.inl>
}

#endif
