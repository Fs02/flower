#ifndef FLOWER_MOMENTUM_H
#define FLOWER_MOMENTUM_H

#include <flower/optimizer.h>

namespace flower
{
    class Momentum : public IOptimizerDef
    {
    public:
        Momentum(double lr = 0.01, double mu = 0.01);

        inline const char *type() const { return "Momentum"; }

        inline double lr() const { return lr_; }
        inline double mu() const { return mu_; }

    protected:
        optimizer_ptr create(Net *net) const;

        double lr_;
        double mu_;
    };

    class MomentumOptimizer : public IOptimizer
    {
    public:
        explicit MomentumOptimizer(Net *net, const Momentum &definition);

        inline const char *type() const { return "Momentum"; }

        Eigen::Tensor<double, 2> optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative);

    protected:
        double lr_;
        double mu_;
        Eigen::Tensor<double, 2> vel_;
    };
}

#endif
