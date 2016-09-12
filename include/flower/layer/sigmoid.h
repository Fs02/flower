#ifndef FLOWER_SIGMOID_H
#define FLOWER_SIGMOID_H

#include <flower/layer.h>
#include <flower/feature.h>
#include <cmath>

namespace flower
{
    class Feature;

    class SigmoidDef : public ILayerDef
    {
    public:
        SigmoidDef(unsigned int size);

        inline const char *type() const { return "Sigmoid"; }

        unsigned int size_;
    };

    class Sigmoid : ILayer
    {
    public:
        explicit Sigmoid(Net* net, const char *name, SigmoidDef *definition);

        inline const char *type() const { return "Sigmoid"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);
    };
}

#endif
