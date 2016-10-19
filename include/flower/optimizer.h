#ifndef FLOWER_OPTIMIZER_H
#define FLOWER_OPTIMIZER_H

#include <flower/tensor.h>
#include <memory>

namespace flower {
    template<typename Scalar> class Net;
    template<typename Scalar> class IOptimizer;
    template<typename Scalar> class IOptimizerOp;

    template<typename Scalar>
    using OptimizerPtr = std::shared_ptr<IOptimizerOp<Scalar>>;

    template<typename Scalar>
    class IOptimizer
    {
    public:
        IOptimizer();

        virtual inline const char *type() const = 0;

        virtual OptimizerPtr<Scalar> create(Net<Scalar> *net) const = 0;
    };

    template<typename Scalar>
    class IOptimizerOp
    {
    public:
        IOptimizerOp() = delete;
        IOptimizerOp(const IOptimizerOp<Scalar>&) = delete;

        explicit IOptimizerOp(Net<Scalar>* net, const IOptimizer<Scalar> &definition);

        virtual inline const char *type() const = 0;

        virtual Tensor<Scalar, 2> optimize(const Tensor<Scalar, 2> &weight, const Tensor<Scalar, 2> &derivative) = 0;
        virtual Tensor<Scalar, 4> optimize(const Tensor<Scalar, 4> &weight, const Tensor<Scalar, 4> &derivative) = 0;

    protected:
        Net<Scalar>* net_;
    };

    #include <flower/optimizer.inl>
}

#endif
