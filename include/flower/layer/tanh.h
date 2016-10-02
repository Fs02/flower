#ifndef FLOWER_TANH_H
#define FLOWER_TANH_H

#include <flower/layer.h>

namespace flower
{
    class TanhDef : public ILayerDef
    {
    public:
        TanhDef();

        inline const char *type() const { return "Tanh"; }

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Tanh : public ILayer
    {
    public:
        explicit Tanh(Net *net, const char *name, const TanhDef &definition);

        inline const char *type() const { return "Tanh"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data, bool train = false);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

    protected:
        Eigen::MatrixXd data_;

        Eigen::Tensor<double, 2> input_;
    };
}

#endif
