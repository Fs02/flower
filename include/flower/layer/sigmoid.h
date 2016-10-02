#ifndef FLOWER_SIGMOID_H
#define FLOWER_SIGMOID_H

#include <flower/layer.h>

namespace flower
{
    class Sigmoid : public ILayerDef
    {
    public:
        Sigmoid();

        inline const char *type() const { return "Sigmoid"; }

        unsigned int size_;

    protected:
        layer_ptr create(Net *net) const;
    };

    class SigmoidLayer : public ILayer
    {
    public:
        explicit SigmoidLayer(Net *net, const Sigmoid &definition);

        inline const char *type() const { return "Sigmoid"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

    protected:
        Eigen::Tensor<double, 2> data_;
    };
}

#endif
