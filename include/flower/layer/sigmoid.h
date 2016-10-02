#ifndef FLOWER_SIGMOID_H
#define FLOWER_SIGMOID_H

#include <flower/layer.h>

namespace flower
{
    class SigmoidDef : public ILayerDef
    {
    public:
        SigmoidDef();

        inline const char *type() const { return "Sigmoid"; }

        unsigned int size_;

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Sigmoid : public ILayer
    {
    public:
        explicit Sigmoid(Net *net, const char *name, const SigmoidDef &definition);

        inline const char *type() const { return "Sigmoid"; }

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
