#ifndef FLOWER_ELU_H
#define FLOWER_ELU_H

#include <flower/layer.h>

namespace flower
{
    class Elu : public ILayerDef
    {
    public:
        Elu(double alpha = 1.0);

        inline const char *type() const { return "Elu"; }

        inline double alpha() const { return alpha_; }

        double alpha_;

    protected:
        layer_ptr create(Net *net) const;
    };

    class EluLayer : public ILayer
    {
    public:
        explicit EluLayer(Net *net, const Elu &definition);

        inline const char *type() const { return "Elu"; }

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

    protected:
        Eigen::Tensor<double, 2> data_;
        double alpha_;
    };
}

#endif
