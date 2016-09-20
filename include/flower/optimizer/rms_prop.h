#ifndef FLOWER_RMS_PROP_H
#define FLOWER_RMS_PROP_H

#include <flower/optimizer.h>

namespace flower
{
    class RmsPropDef : public IOptimizerDef
    {
    public:
        RmsPropDef(double learning_rate, double decay_rate);

        inline const char *type() const { return "RmsProp"; }

        inline double learning_rate() const { return learning_rate_; }
        inline double decay_rate() const { return decay_rate_; }

    protected:
        optimizer_ptr create(Net *net) const;

        double learning_rate_;
        double decay_rate_;
    };

    class RmsProp : public IOptimizer
    {
    public:
        explicit RmsProp(Net *net, const RmsPropDef &definition);

        inline const char *type() const { return "RmsProp"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &dw);

    protected:
        double learning_rate_;
        double decay_rate_;
        Eigen::ArrayXXd  gt_;
    };
}

#endif
