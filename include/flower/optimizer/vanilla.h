#ifndef FLOWER_VANILLA_H
#define FLOWER_VANILLA_H

#include <flower/optimizer.h>

namespace flower
{
    class VanillaDef : public IOptimizerDef
    {
    public:
        VanillaDef(double lr = 0.01);

        inline const char *type() const { return "StochasticGradientDescent"; }

        inline double lr() const { return lr_; }

    protected:
        optimizer_ptr create(Net *net) const;
        static optimizer_ptr instance_;

        double lr_;
    };

    class Vanilla : public IOptimizer
    {
    public:
        explicit Vanilla(Net *net, const VanillaDef &definition);

        inline const char *type() const { return "StochasticGradientDescent"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &derivative);

        Eigen::Tensor<double, 2> optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative);

    protected:
        double lr_;
    };
}

#endif
