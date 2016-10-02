#ifndef FLOWER_VANILLA_H
#define FLOWER_VANILLA_H

#include <flower/optimizer.h>

namespace flower
{
    class Vanilla : public IOptimizerDef
    {
    public:
        Vanilla(double lr = 0.01);

        inline const char *type() const { return "Vanilla"; }

        inline double lr() const { return lr_; }

    protected:
        optimizer_ptr create(Net *net) const;
        static optimizer_ptr instance_;

        double lr_;
    };

    class VanillaOptimizer : public IOptimizer
    {
    public:
        explicit VanillaOptimizer(Net *net, const Vanilla &definition);

        inline const char *type() const { return "Vanilla"; }

        Eigen::Tensor<double, 2> optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative);

    protected:
        double lr_;
    };
}

#endif
