#ifndef FLOWER_MOMENTUM_H
#define FLOWER_MOMENTUM_H

#include <flower/optimizer.h>

namespace flower
{
    class MomentumDef : public IOptimizerDef
    {
    public:
        MomentumDef(double lr = 0.01, double mu = 0.01);

        inline const char *type() const { return "Momentum"; }

        inline double lr() const { return lr_; }
        inline double mu() const { return mu_; }

    protected:
        optimizer_ptr create(Net *net) const;

        double lr_;
        double mu_;
    };

    class Momentum : public IOptimizer
    {
    public:
        explicit Momentum(Net *net, const MomentumDef &definition);

        inline const char *type() const { return "Momentum"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &derivative);

    protected:
        double lr_;
        double mu_;
        Eigen::MatrixXd  velocity_;
    };
}

#endif
