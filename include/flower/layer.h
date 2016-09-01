#ifndef ACTIVATION_H
#define ACTIVATION_H

namespace flower {
    class Feature;

    class Layer
    {
    public:
        virtual inline const char *type() const = 0;

        virtual void forward(Feature &bottom, Feature &top) = 0;
        virtual void backward(Feature &top, Feature &bottom) = 0;
    };
}

#endif
