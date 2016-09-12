#ifndef FLOWER_LAYER_H
#define FLOWER_LAYER_H

namespace flower {
    class Net;
    class Feature;
    class ILayer;

    class ILayerDef
    {
    public:
        ILayerDef();

        virtual inline const char *type() const = 0;

        virtual ILayer *create(Net *net, const char* name) = 0;
    };

    class ILayer
    {
    public:
        ILayer() = delete;
        ILayer(const ILayer&) = delete;

        explicit ILayer(Net* net, const char *name, ILayerDef *definition);

        virtual inline const char *type() const = 0;
        virtual inline const char *name() const { return name_; }

        virtual void forward(Feature &bottom, Feature &top) = 0;
        virtual void backward(Feature &top, Feature &bottom) = 0;

    protected:
        const char *name_;
        Net* net_;
    };
}

#endif
