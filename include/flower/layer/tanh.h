#ifndef FLOWER_TANH_H
#define FLOWER_TANH_H

#include <flower/layer.h>

namespace flower
{
    class Tanh : public ILayerDef
    {
    public:
        Tanh();

        inline const char *type() const { return "Tanh"; }

    protected:
        layer_ptr create(Net *net) const;
    };

    class TanhLayer : public ILayer
    {
    public:
        explicit TanhLayer(Net *net, const Tanh &definition);

        inline const char *type() const { return "Tanh"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

    protected:
        Eigen::Tensor<double, 2> data_;
    };
}

#endif
