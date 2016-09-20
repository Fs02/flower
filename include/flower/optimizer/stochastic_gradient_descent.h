#ifndef FLOWER_STOCHASTIC_GRADIENT_DESCENT_H
#define FLOWER_STOCHASTIC_GRADIENT_DESCENT_H

#include <flower/optimizer.h>

namespace flower
{
    class StochasticGradientDescentDef : public IOptimizerDef
    {
    public:
        StochasticGradientDescentDef(double learning_rate);

        inline const char *type() const { return "StochasticGradientDescent"; }

        inline double learning_rate() const { return learning_rate_; }

    protected:
        optimizer_ptr create(Net *net) const;
        static optimizer_ptr instance_;

        double learning_rate_;
    };

    class StochasticGradientDescent : public IOptimizer
    {
    public:
        explicit StochasticGradientDescent(Net *net, const StochasticGradientDescentDef &definition);

        inline const char *type() const { return "StochasticGradientDescent"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw);

    protected:
        double learning_rate_;
    };
}

#endif
