#ifndef FLOWER_ADAM_H
#define FLOWER_ADAM_H

#include <flower/optimizer.h>

namespace flower
{
    class Adam : public IOptimizerDef
    {
    public:
        Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-08);

        inline const char *type() const { return "Adam"; }

        inline double lr() const { return lr_; }
        inline double beta1() const { return beta1_; }
        inline double beta2() const { return beta2_; }
        inline double eps() const { return eps_; }

    protected:
        optimizer_ptr create(Net *net) const;

        double lr_;
        double beta1_;
        double beta2_;
        double eps_;
    };

    class AdamOptimizer : public IOptimizer
    {
    public:
        explicit AdamOptimizer(Net *net, const Adam &definition);

        inline const char *type() const { return "Adam"; }

        Tensor<double, 2, RowMajor> optimize(const Tensor<double, 2, RowMajor> &weight, const Tensor<double, 2, RowMajor> &derivative);

    protected:
        double lr_;
        double beta1_;
        double beta2_;
        double eps_;
        Tensor<double, 2, RowMajor>  m_;
        Tensor<double, 2, RowMajor>  v_;
    };
}

#endif
