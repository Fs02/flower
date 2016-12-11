#ifndef FLOWER_BATCH_NORMALIZATION_H
#define FLOWER_BATCH_NORMALIZATION_H

#include <flower/layer.h>

namespace flower
{
    class BatchNormalizationDef : public ILayerDef
    {
    public:
        BatchNormalizationDef();

        inline const char *type() const { return "BatchNormalization"; }

    protected:
        layer_ptr create(Net *net) const;
    };

    class BatchNormalization : public ILayer
    {
    public:
        explicit BatchNormalization(Net *net, const BatchNormalizationDef &definition);

        inline const char *type() const { return "BatchNormalization"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);
    };
}

#endif
