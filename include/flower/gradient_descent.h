#ifndef FLOWER_GRADIENT_DESCENT_H
#define FLOWER_GRADIENT_DESCENT_H

#include <Eigen/Core>
#include <vector>

namespace flower {
    class Net;
    class IOptimizerDef;

    class GradientDescent
    {
    public:
        GradientDescent(Net *net, const IOptimizerDef& optimizer_def);
        ~GradientDescent();

        double feed(const Eigen::MatrixXd &data, const Eigen::MatrixXd &target);

    private:
        Net *net_;
    };
}

#endif
