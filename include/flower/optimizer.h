#ifndef FLOWER_OPTIMIZER_H
#define FLOWER_OPTIMIZER_H

#include <Eigen/CXX11/Tensor>
#include <memory>

namespace flower {
    class Net;
    class IOptimizer;

    typedef std::shared_ptr<IOptimizer> optimizer_ptr;

    class IOptimizerDef
    {
    public:
        IOptimizerDef();

        virtual inline const char *type() const = 0;

        virtual optimizer_ptr create(Net *net) const = 0;
    };

    class IOptimizer
    {
    public:
        IOptimizer() = delete;
        IOptimizer(const IOptimizer&) = delete;

        explicit IOptimizer(Net* net, const IOptimizerDef &definition);

        virtual inline const char *type() const = 0;

        virtual Eigen::Tensor<double, 2> optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative) = 0;

    protected:
        Net* net_;
    };
}

#endif
