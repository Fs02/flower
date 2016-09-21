#ifndef FLOWER_STOCHASTIC_GRADIENT_DESCENT_H
#define FLOWER_STOCHASTIC_GRADIENT_DESCENT_H

#include <flower/optimizer.h>

namespace flower
{
    class StochasticGradientDescentDef : public IOptimizerDef
    {
    public:
        StochasticGradientDescentDef(double lr = 0.01);

        inline const char *type() const { return "StochasticGradientDescent"; }

        inline double lr() const { return lr_; }

    protected:
        optimizer_ptr create(Net *net) const;
        static optimizer_ptr instance_;

        double lr_;
    };

    class StochasticGradientDescent : public IOptimizer
    {
    public:
        explicit StochasticGradientDescent(Net *net, const StochasticGradientDescentDef &definition);

        inline const char *type() const { return "StochasticGradientDescent"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &derivative);

    protected:
        double lr_;
    };
}

#endif
