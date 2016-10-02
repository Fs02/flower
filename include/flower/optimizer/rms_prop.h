#ifndef FLOWER_RMS_PROP_H
#define FLOWER_RMS_PROP_H

#include <flower/optimizer.h>

namespace flower
{
    class RmsProp : public IOptimizerDef
    {
    public:
        RmsProp(double lr = 0.001, double decay = 0.0, double eps = 1e-08);

        inline const char *type() const { return "RmsProp"; }

        inline double lr() const { return lr_; }
        inline double decay() const { return decay_; }
        inline double eps() const { return eps_; }

    protected:
        optimizer_ptr create(Net *net) const;

        double lr_;
        double decay_;
        double eps_;
    };

    class RmsPropOptimizer : public IOptimizer
    {
    public:
        explicit RmsPropOptimizer(Net *net, const RmsProp &definition);

        inline const char *type() const { return "RmsProp"; }

        Eigen::Tensor<double, 2> optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative);

    protected:
        double lr_;
        double decay_;
        double eps_;
        Eigen::Tensor<double, 2> gt_;
    };
}

#endif
