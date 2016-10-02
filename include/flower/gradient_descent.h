#ifndef FLOWER_GRADIENT_DESCENT_H
#define FLOWER_GRADIENT_DESCENT_H

#include <Eigen/CXX11/Tensor>
#include <vector>

namespace flower {
    class Net;
    class IOptimizerDef;

    class GradientDescent
    {
    public:
        GradientDescent(Net *net, double learning_rate);
        GradientDescent(Net *net, const IOptimizerDef& optimizer_def);
        ~GradientDescent();

        Eigen::Tensor<double, 0> feed(const Eigen::Tensor<double, 2> &data, const Eigen::Tensor<double, 2> &target);

    private:
        Net *net_;
    };
}

#endif
