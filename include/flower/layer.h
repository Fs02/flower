#ifndef ACTIVATION_H
#define ACTIVATION_H

namespace flower {
    class Blob;

    class Layer
    {
    public:
        virtual inline const char* type() const = 0;

        virtual void forward(Blob& bottom, Blob& top) = 0;
        virtual void backward(Blob& top, Blob& bottom) = 0;
    };
}

#endif
