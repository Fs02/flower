#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <flower/layer.h>
#include <flower/feature.h>

namespace flower
{
    class FullyConnectedDef : public ILayerDef
    {
    public:
        FullyConnectedDef(const char *name, unsigned int bottom_size, unsigned int top_size);

        inline const char *type() const { return "FullyConnected"; }
        inline unsigned int bottom_size() const { return bottom_size_; }
        inline unsigned int top_size() const { return top_size_; }

    protected:
        unsigned int bottom_size_;
        unsigned int top_size_;
    };

    class FullyConnected : ILayer
    {
    public:
        FullyConnected(FullyConnectedDef *definition);

        inline const char *type() const { return "FullyConnected"; }

        void setup(ILayerDef *definition);

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);

    protected:
        Feature weights_;
        Feature bias_;
    };
}

#endif
