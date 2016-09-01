#ifndef ACTIVATION_H
#define ACTIVATION_H

namespace flower {
    class Feature;

    class ILayerDef
    {
    public:
        ILayerDef() = delete;
        ILayerDef(const char *name);

        virtual inline const char *type() const = 0;

        const inline char *name() const { return name_; }

    protected:
        const char *name_;
    };

    class ILayer
    {
    public:
        ILayer(ILayerDef *definition);
        ILayer(const ILayer&) = delete;

        virtual inline const char *type() const = 0;
        virtual inline const char *name() const { return name_; }

        virtual void forward(Feature &bottom, Feature &top) = 0;
        virtual void backward(Feature &top, Feature &bottom) = 0;

    protected:
        const char *name_;
    };
}

#endif
