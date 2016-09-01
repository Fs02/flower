#ifndef SIGMOID_H
#define SIGMOID_H

#include <flower/layer.h>
#include <cmath>

namespace flower
{
    class Feature;

    class Sigmoid : Layer
    {
    public:
        Sigmoid();

        inline const char *type() const { return "Sigmoid"; }

        void forward(Feature &bottom, Feature &top);
        void backward(Feature &top, Feature &bottom);
    };
}

#endif
