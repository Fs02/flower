#ifndef FLOWER_RMS_PROP_H
#define FLOWER_RMS_PROP_H

#include <flower/optimizer.h>

namespace flower
{
    class RmsPropDef : public IOptimizerDef
    {
    public:
        RmsPropDef(double lr = 0.001, double decay = 0.0, double eps = 1e-08);

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

    class RmsProp : public IOptimizer
    {
    public:
        explicit RmsProp(Net *net, const RmsPropDef &definition);

        inline const char *type() const { return "RmsProp"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw);

    protected:
        double lr_;
        double decay_;
        double eps_;
        Eigen::ArrayXXd  gt_;
    };
}

#endif
