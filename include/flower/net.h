#ifndef FLOWER_NET_H
#define FLOWER_NET_H

#include <flower/layer.h>
#include <flower/optimizer.h>
#include <Eigen/Core>
#include <vector>

namespace flower {
    class SupervisedLearning;

    class Net
    {
        friend class SupervisedLearning;
    public:
        Net();
        ~Net();

        Eigen::MatrixXd infer(const Eigen::MatrixXd &data) const;

        void add(const char *name, const ILayerDef &definition);

        const std::vector<std::pair<const char*, layer_ptr>> &layers() const;
        inline int epoch() const { return epoch_; }

    private:
        std::vector<std::pair<const char*, layer_ptr>> layers_;
        int epoch_;
    };
}

#endif
