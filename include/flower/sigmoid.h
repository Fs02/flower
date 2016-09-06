#ifndef FLOWER_SIGMOID_H
#define FLOWER_SIGMOID_H

#include <flower/layer.h>
#include <cmath>

namespace flower
{
    class Feature;

    class SigmoidDef : public ILayerDef
    {
    public:
        SigmoidDef(const char *name, unsigned int size);

        inline const char *type() const { return "Sigmoid"; }
        inline unsigned int size() const { return size_; }

    protected:
        unsigned int size_;
    };

    class Sigmoid : ILayer
    {
    public:
        Sigmoid(SigmoidDef *definition);

        inline const char *type() const { return "Sigmoid"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);
    };
}

#endif
