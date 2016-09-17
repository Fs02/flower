#ifndef FLOWER_SOFTMAX_LOSS_H
#define FLOWER_SOFTMAX_LOSS_H

#include <flower/net.h>
#include <flower/layer.h>
#include <flower/feature.h>

namespace flower
{
    class SoftmaxLossDef : public ILayerDef
    {
    public:
        SoftmaxLossDef();

        inline const char *type() const { return "SoftmaxLoss"; }

        layer_ptr create(Net *net, const char* name) const;
    };

    class SoftmaxLoss : public ILayer
    {
    public:
        SoftmaxLoss(Net *net, const char *name, const SoftmaxLossDef &definition);

        inline const char *type() const { return "SoftmaxLoss"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);

        Eigen::MatrixXd forward(const Eigen::MatrixXd &bottom_feat);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &top_diff);
    };
}

#endif
