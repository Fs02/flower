#ifndef FLOWER_MOMENTUM_H
#define FLOWER_MOMENTUM_H

#include <flower/optimizer.h>

namespace flower
{
    class MomentumDef : public IOptimizerDef
    {
    public:
        MomentumDef(double learning_rate, double mu);

        inline const char *type() const { return "Momentum"; }

        inline double learning_rate() const { return learning_rate_; }
        inline double mu() const { return mu_; }

    protected:
        optimizer_ptr create(Net *net) const;

        double learning_rate_;
        double mu_;
    };

    class Momentum : public IOptimizer
    {
    public:
        explicit Momentum(Net *net, const MomentumDef &definition);

        inline const char *type() const { return "Momentum"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw);

    protected:
        double learning_rate_;
        double mu_;
        Eigen::MatrixXd  velocity_;
    };
}

#endif
