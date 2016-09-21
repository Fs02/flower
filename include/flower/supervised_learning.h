#ifndef FLOWER_SUPERVISED_LEARNING_H
#define FLOWER_SUPERVISED_LEARNING_H

#include <Eigen/Core>
#include <vector>

namespace flower {
    class Net;
    class IOptimizerDef;

    class SupervisedLearning
    {
    public:
        SupervisedLearning(Net *net, const IOptimizerDef& optimizer_def);
        ~SupervisedLearning();

        double feed(const Eigen::MatrixXd &data, const Eigen::MatrixXd &target);

    private:
        Net *net_;
    };
}

#endif
