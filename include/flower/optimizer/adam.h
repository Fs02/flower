#ifndef FLOWER_ADAM_H
#define FLOWER_ADAM_H

#include <flower/optimizer.h>

namespace flower
{
    class AdamDef : public IOptimizerDef
    {
    public:
        AdamDef(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-08);

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

    class Adam : public IOptimizer
    {
    public:
        explicit Adam(Net *net, const AdamDef &definition);

        inline const char *type() const { return "Adam"; }

        Eigen::MatrixXd optimize(const Eigen::MatrixXd &weight, const Eigen::MatrixXd &derivative);

        Eigen::Tensor<double, 2> optimize(const Eigen::Tensor<double, 2> &weight, const Eigen::Tensor<double, 2> &derivative);

    protected:
        double lr_;
        double beta1_;
        double beta2_;
        double eps_;
        Eigen::ArrayXXd  m_;
        Eigen::ArrayXXd  v_;

        Eigen::Tensor<double, 2>  t_m_;
        Eigen::Tensor<double, 2>  t_v_;
    };
}

#endif
