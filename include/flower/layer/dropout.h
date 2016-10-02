#ifndef FLOWER_DROPOUT_H
#define FLOWER_DROPOUT_H

#include <flower/layer.h>

namespace flower
{
    class DropoutDef : public ILayerDef
    {
    public:
        DropoutDef(double probability = 0.5);

        inline const char *type() const { return "Dropout"; }

        inline double probability() const { return probability_; }

        double probability_;

    protected:
        layer_ptr create(Net *net, const char* name) const;
    };

    class Dropout : public ILayer
    {
    public:
        explicit Dropout(Net *net, const char *name, const DropoutDef &definition);

        inline const char *type() const { return "Dropout"; }

        Eigen::MatrixXd forward(const Eigen::MatrixXd &data, bool train = false);
        Eigen::MatrixXd backward(const Eigen::MatrixXd &errors);

        Eigen::Tensor<double, 2> forward(const Eigen::Tensor<double, 2> &data, bool train = false);
        Eigen::Tensor<double, 2> backward(const Eigen::Tensor<double, 2> &errors);

    protected:
        Eigen::ArrayXXd mask_;
        double probability_;
    };
}

#endif
