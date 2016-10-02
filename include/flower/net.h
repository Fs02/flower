#ifndef FLOWER_NET_H
#define FLOWER_NET_H

#include <flower/layer.h>
#include <flower/optimizer.h>
#include <vector>

namespace flower {
    class Net
    {
        friend class GradientDescent;
    public:
        Net();
        ~Net();

        Eigen::Tensor<double, 2> infer(const Eigen::Tensor<double, 2> &data) const;

        void add(const ILayerDef &definition);

        const std::vector<layer_ptr> &layers() const;
        inline int epoch() const { return epoch_; }

    private:
        std::vector<layer_ptr> layers_;
        int epoch_;
    };
}

#endif
