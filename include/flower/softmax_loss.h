#ifndef FLOWER_SOFTMAX_LOSS_H
#define FLOWER_SOFTMAX_LOSS_H

#include <flower/layer.h>
#include <flower/feature.h>

namespace flower
{
    class SoftmaxLossDef : public ILayerDef
    {
    public:
        SoftmaxLossDef(const char *name);

        inline const char *type() const { return "SoftmaxLoss"; }
    };

    class SoftmaxLoss : ILayer
    {
    public:
        SoftmaxLoss(SoftmaxLossDef *definition);

        inline const char *type() const { return "SoftmaxLoss"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);
    };
}

#endif
