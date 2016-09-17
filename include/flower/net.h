#ifndef FLOWER_NET_H
#define FLOWER_NET_H

#include <flower/layer.h>
#include <Eigen/Core>
#include <vector>

namespace flower {
    class Net
    {
    public:
        Net();
        ~Net();

        double train(const Eigen::MatrixXd& data, const Eigen::MatrixXd& target);
        double eval();

        void add(const char *name, const ILayerDef& definition);

    private:
        std::vector<std::pair<const char*, layer_ptr>> layers_;
    };
}

#endif
