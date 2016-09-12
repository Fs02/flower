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

        ILayer *create(Net *net, const char* name);
    };

    class SoftmaxLoss : public ILayer
    {
    public:
        SoftmaxLoss(Net *net, const char *name, SoftmaxLossDef *definition);

        inline const char *type() const { return "SoftmaxLoss"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);
    };
}

#endif
