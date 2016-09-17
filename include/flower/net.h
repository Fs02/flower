#ifndef FLOWER_NET_H
#define FLOWER_NET_H

#include <flower/layer.h>
#include <flower/feature.h>
#include <Eigen/Core>
#include <vector>

namespace flower {
    class Net
    {
    public:
        Net();
        ~Net();

        void train(const Eigen::MatrixXd& data, const Eigen::MatrixXd& label);
        void eval();

        void add(const char *name, const ILayerDef& definition);

    private:
        std::vector<std::pair<const char*, layer_ptr>> layers_;
    };
}

#endif
