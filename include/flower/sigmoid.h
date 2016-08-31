#ifndef SIGMOID_H
#define SIGMOID_H

#include <flower/layer.h>
#include <cmath>

namespace flower
{
    class Blob;

    class Sigmoid : Layer
    {
    public:
        Sigmoid();

        inline const char* type() const { return "Sigmoid"; }

        void forward(Blob& bottom, Blob& top);
        void backward(Blob& top, Blob& bottom);
    };
}

#endif
