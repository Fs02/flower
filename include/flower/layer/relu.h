#ifndef FLOWER_RELU_H
#define FLOWER_RELU_H

#include <flower/layer.h>

namespace flower
{
    class ReluDef : public ILayerDef
    {
    public:
        ReluDef();

        inline const char *type() const { return "Relu"; }

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Relu : public ILayer
    {
    public:
        explicit Relu(Net *net, const char *name, const ReluDef &definition);

        inline const char *type() const { return "Relu"; }

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
