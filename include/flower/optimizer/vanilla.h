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

        Tensor<Scalar, 2, RowMajor> optimize(const Tensor<Scalar, 2, RowMajor> &weight, const Tensor<Scalar, 2, RowMajor> &derivative);
        Tensor<Scalar, 4, RowMajor> optimize(const Tensor<Scalar, 4, RowMajor> &weight, const Tensor<Scalar, 4, RowMajor> &derivative);

        template<int rank>
        Tensor<Scalar, rank, RowMajor> compute(const Tensor<Scalar, rank, RowMajor> &weight, const Tensor<Scalar, rank, RowMajor> &derivative);

    protected:
        double lr_;
    };

    #include <flower/optimizer/vanilla.inl>
}

#endif
