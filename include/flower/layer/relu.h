#ifndef FLOWER_RELU_H
#define FLOWER_RELU_H

#include <flower/layer.h>

namespace flower
{
    class Relu : public ILayerDef
    {
    public:
        Relu();

        inline const char *type() const { return "Relu"; }

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class ReluLayer : public ILayer
    {
    public:
        explicit ReluLayer(Net *net, const char *name, const Relu &definition);

        inline const char *type() const { return "Relu"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

    protected:
        Eigen::Tensor<double, 2> data_;
    };
}

#endif
